import logging
import queue
import threading

from datetime import datetime, timedelta
from typing import Callable, Iterable, Any, List, Tuple, Union

logger = logging.getLogger(__name__)


class ProcessingThreadPool:
    """A simple thread pool for running worker functions with different argument patterns."""
    
    def __init__(self, num_workers: int, thread_name_prefix: str = "Worker"):
        self.num_workers = num_workers
        self.thread_name_prefix = thread_name_prefix
        self.threads: List[threading.Thread] = []
    
    def run_workers(self, worker_func: Callable, worker_args_list: List[Tuple]) -> None:
        """
        Run worker functions in parallel threads and wait for completion.
        
        Args:
            worker_func: The function to run in each thread
            worker_args_list: List of argument tuples, one per worker
        """
        logger.info(f"Starting {len(worker_args_list)} {self.thread_name_prefix} threads...")
        
        # Clear any previous threads
        self.threads.clear()
        
        # Create and start threads
        for worker_id, args in enumerate(worker_args_list):
            thread = threading.Thread(
                target=worker_func,
                args=args,
                name=f"{self.thread_name_prefix}-{worker_id}"
            )
            self.threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in self.threads:
            try:
                thread.join()
            except Exception as e:
                logger.error(f"Error joining thread {thread.name}: {e}")
        
        logger.info(f"{self.thread_name_prefix} processing complete!")


class ProcessingPipeline:
    """Manages a producer-consumer-writer workflow with multiple worker threads."""

    def __init__(
        self,
        producer: Union[Callable[[], Iterable[Any]], "ProcessingPipeline", None],
        consumer: Callable[[Any], Any],
        writer: Callable[[Any], None] = None,
        num_workers: int = 4,
        queue_size: int = None,
        writer_queue_size: int = None,
        task_name: str = None,
    ):
        """
        Initializes the processing pipeline.

        Args:
            producer: Either:
                     - A no-argument callable called as producer() that returns an
                       iterable of input items (Iterable[Any]), or
                     - Another ProcessingPipeline instance, in which case this
                       pipeline will consume items produced by the upstream
                       pipeline.
            consumer: A one-argument callable called as consumer(item) that processes
                      a single input item.
                      - If writer is provided, consumer(item) should return the
                        object that will be passed to writer.
                      - If writer is not provided, the return value is ignored.
            writer: An optional one-argument callable called as writer(result) that
                    consumes the value returned by consumer and performs side
                    effects (e.g., writing to disk). It should return None.
            num_workers: The number of concurrent consumer threads.
            queue_size: Max size of the task queue. Defaults to num_workers * 4.
            writer_queue_size: Max size of the writer queue. Defaults to queue_size.
            task_name: Optional descriptive name for the task (e.g., "Keogram Creation").
                       Used in log messages for better clarity.
        """
        # Check if producer is a pipeline or callable
        is_pipeline_producer = isinstance(producer, ProcessingPipeline)
        if not is_pipeline_producer and producer is not None and not callable(producer):
            raise TypeError("Producer must be a callable that returns an iterable, or a ProcessingPipeline.")
        if not callable(consumer):
            raise TypeError("Consumer must be a callable.")
        if writer and not callable(writer):
            raise TypeError("Writer must be a callable if provided.")
        
        # Validate that upstream pipeline has a writer
        if is_pipeline_producer and not producer.writer:
            raise ValueError("Upstream ProcessingPipeline must have a writer to feed this pipeline.")

        self.producer = producer
        self.consumer = consumer
        self.writer = writer
        self.num_workers = num_workers
        self.task_name = task_name

        if queue_size is None:
            queue_size = num_workers * 4
        self.task_queue = queue.Queue(maxsize=queue_size)

        # For monitoring
        self._stop_event = threading.Event()
        self._task_queue_samples = []
        self._task_queue_empty_time = 0
        self._processing_time = 0

        # Writer-specific setup
        self.writer_queue = None
        if self.writer:
            if writer_queue_size is None:
                writer_queue_size = queue_size
            self.writer_queue = queue.Queue(maxsize=writer_queue_size)
            self._writer_queue_samples = []
            self._writer_queue_empty_time = 0

    def _producer_thread(self):
        """Internal method to run the producer and populate the task queue."""
        task_prefix = f"{self.task_name}->" if self.task_name else ""
        logger.info(f"--- Starting producer thread (target: {task_prefix}{self.producer.__name__}) ---")
        try:
            for item in self.producer():
                self.task_queue.put(item)
        finally:
            logger.info(f"--- Producer thread ({task_prefix}{self.producer.__name__}) finished. All tasks have been queued. ---")

    def _consumer_thread(self, thread_num: int):
        """Internal method for consumer workers."""
        task_prefix = f"{self.task_name}->" if self.task_name else ""
        logger.info(f"--- Consumer thread {thread_num}/{self.num_workers} started (target: {task_prefix}{self.consumer.__name__}) ---")
        while not (self._stop_event.is_set() and self.task_queue.empty()):
            try:
                task = self.task_queue.get(timeout=0.1)
                if task is None:  # Sentinel value
                    self.task_queue.task_done()
                    break

                try:
                    result = self.consumer(task)

                    # If there's a writer, pass the result to the writer queue
                    if self.writer and result is not None:
                        self.writer_queue.put(result)
                except Exception as e:
                    logger.error(f"Exception in consumer thread {thread_num} processing task: {e}", exc_info=True)
                    # Continue processing other tasks even if one fails
                finally:
                    # Always mark task as done to prevent queue.join() from hanging
                    self.task_queue.task_done()
            except queue.Empty:
                continue

    def _writer_thread(self):
        """Internal method for the writer thread. Only runs if a writer is configured."""
        task_prefix = f"{self.task_name}->" if self.task_name else ""
        logger.info(f"--- Starting writer thread (target: {task_prefix}{self.writer.__name__}) ---")
        while not (self._stop_event.is_set() and self.writer_queue.empty()):
            try:
                item_to_write = self.writer_queue.get(timeout=0.1)
                if item_to_write is None:  # Sentinel
                    self.writer_queue.task_done()
                    break

                self.writer(item_to_write)
                self.writer_queue.task_done()
            except queue.Empty:
                continue

        logger.info(f"--- Writer thread ({task_prefix}{self.writer.__name__}) finished. ---")

    def _monitor_queues(self, interval=0.1):
        """Monitors the queue sizes at regular intervals."""
        while not self._stop_event.wait(interval):
            task_qsize = self.task_queue.qsize()
            self._task_queue_samples.append(task_qsize)
            if task_qsize == 0:
                self._task_queue_empty_time += interval

            if self.writer_queue:
                writer_qsize = self.writer_queue.qsize()
                self._writer_queue_samples.append(writer_qsize)
                if writer_qsize == 0:
                    self._writer_queue_empty_time += interval

    def run(self):
        """Starts and runs the entire processing pipeline.
        
        If producer is a ProcessingPipeline, both pipelines run in parallel with
        the upstream pipeline enqueuing its consumer results into this pipeline's
        task queue (while still running its own writer side effects).
        """
        import time
        start_time = time.time()
        
        task_desc = f" ({self.task_name})" if self.task_name else ""
        logger.info(f"Starting pipeline{task_desc} with {self.num_workers} worker threads...")
        worker_threads = []
        writer_thread = None
        producer_thread = None
        upstream_pipeline = None

        # Check if producer is a pipeline
        if isinstance(self.producer, ProcessingPipeline):
            # Producer is an upstream pipeline - wrap its writer to feed our queue
            upstream_pipeline = self.producer
            original_writer = upstream_pipeline.writer
            
            def feeding_writer(result):
                if result is None:
                    return
                # Preserve upstream writer side effects (e.g., writing to disk)
                if original_writer is not None:
                    original_writer(result)
                # Feed the downstream pipeline with the same produced result
                self.task_queue.put(result)
            
            upstream_pipeline.writer = feeding_writer
            
            # Start upstream pipeline in separate thread
            producer_thread = threading.Thread(target=upstream_pipeline.run)
            producer_thread.start()
        elif self.producer is not None:
            # Producer is a regular callable - start producer thread
            producer_thread = threading.Thread(target=self._producer_thread)
            producer_thread.start()

        monitor_thread = threading.Thread(target=self._monitor_queues)
        monitor_thread.start()

        # Start the writer thread if configured
        if self.writer:
            writer_thread = threading.Thread(target=self._writer_thread)
            writer_thread.start()

        # Start consumer threads
        for i in range(self.num_workers):
            thread = threading.Thread(
                target=self._consumer_thread,
                args=(i + 1,),
                name=f"ConsumerThread-{i+1}",
            )
            thread.start()
            worker_threads.append(thread)

        # Wait for the producer to finish (if it exists)
        if producer_thread is not None:
            producer_thread.join()

        # Wait for the consumers to process all items
        self.task_queue.join()

        # Signal consumers to stop
        for _ in range(self.num_workers):
            self.task_queue.put(None)  # Sentinel to unblock waiting consumers

        # Wait for writer to finish (if it exists)
        if writer_thread:
            self.writer_queue.join()
            self.writer_queue.put(None)  # Sentinel to unblock writer
            writer_thread.join()

        # Stop monitor and join all threads
        self._stop_event.set()
        for thread in worker_threads:
            thread.join()
        monitor_thread.join()
        
        self._processing_time = time.time() - start_time

        task_desc = f" ({self.task_name})" if self.task_name else ""
        logger.info(f"Pipeline{task_desc} processing complete!")

    def get_queue_stats(self) -> dict:
        """Returns a dictionary with queue statistics."""
        stats = {}
        if not self._task_queue_samples:
            stats["task_queue"] = {"avg_depth": 0, "empty_time": 0}
        else:
            avg_depth = sum(self._task_queue_samples) / len(self._task_queue_samples)
            stats["task_queue"] = {
                "avg_depth": avg_depth,
                "empty_time": self._task_queue_empty_time,
            }

        if self.writer_queue:
            if not self._writer_queue_samples:
                stats["writer_queue"] = {"avg_depth": 0, "empty_time": 0}
            else:
                avg_depth = sum(self._writer_queue_samples) / len(
                    self._writer_queue_samples
                )
                stats["writer_queue"] = {
                    "avg_depth": avg_depth,
                    "empty_time": self._writer_queue_empty_time,
                }
        
        stats["processing_time"] = self._processing_time

        return stats
