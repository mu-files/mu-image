/*
 * VNG (Variable Number of Gradients) Demosaicing Algorithm
 * 
 * Extracted from LibRaw (https://github.com/LibRaw/LibRaw)
 * Originally from dcraw.c by Dave Coffin
 * 
 * Copyright (C) 2008-2024 LibRaw LLC (info@libraw.org)
 * Copyright (C) 1997-2018 Dave Coffin
 * 
 * Licensed under LGPL v2.1 and CDDL v1.0 (dual license)
 * See: https://github.com/LibRaw/LibRaw/blob/master/LICENSE.LGPL
 * 
 * Modifications from original:
 * - Added Python/NumPy C API bindings for use as Python extension
 * - Thread-safety fix: Made 'cp' pointer local instead of static
 * - Added string-based Bayer pattern parsing ("RGGB", "BGGR", "GRBG", "GBRG")
 * - Removed LibRaw-specific dependencies and made standalone
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <string.h>
#include <limits.h>
#include <stdlib.h>

/* Type definitions */
typedef unsigned short ushort;

/* Macros */
#define ABS(x) (((int)(x) ^ ((int)(x) >> 31)) - ((int)(x) >> 31))
#define CLIP(x) ((x) < 0 ? 0 : (x) > 65535 ? 65535 : (x))
#define FORCC for(c=0; c < 3; c++)

/* Bayer pattern lookup */
static unsigned fcol(int row, int col, unsigned filters)
{
  static const char filter[16][16] = {
    {2,1,1,3,2,3,2,0,3,2,3,0,1,2,1,0},
    {0,3,0,2,0,1,3,1,0,1,1,2,0,3,3,2},
    {2,3,3,2,3,1,1,3,3,1,2,1,2,0,0,3},
    {0,1,0,1,0,2,0,2,2,0,3,0,1,3,2,1},
    {3,1,1,2,0,1,0,2,1,3,1,3,0,1,3,0},
    {2,0,0,3,3,2,3,1,2,0,2,0,3,2,2,1},
    {2,3,3,1,2,1,2,1,2,1,1,2,3,0,0,1},
    {1,0,0,2,3,0,0,3,0,3,0,3,2,1,2,3},
    {2,3,3,1,1,2,1,0,3,2,3,0,2,3,1,3},
    {1,0,2,0,3,0,3,2,0,1,1,2,0,1,0,2},
    {0,1,1,3,3,2,2,1,1,3,3,0,2,1,3,2},
    {2,3,2,0,0,1,3,0,2,0,1,2,3,0,1,0},
    {1,3,1,2,3,2,3,2,0,2,0,1,1,0,3,0},
    {0,2,0,3,1,0,0,1,1,3,3,2,3,2,2,1},
    {2,1,3,2,3,1,2,1,0,3,0,2,0,2,0,2},
    {0,3,1,0,0,2,0,3,2,1,3,1,1,3,1,3}
  };
  
  if (filters == 1) return filter[(row+16) & 15][(col+16) & 15];
  if (filters == 9) return (col % 6 + (row % 6) * 6) % 3;
  return (filters >> (((row << 1 & 14) + (col & 1)) << 1) & 3);
}

static void border_interpolate(int border, ushort (*image)[4], int width, int height, unsigned filters)
{
  unsigned row, col, y, x, f, c, sum[8];

  for (row = 0; row < height; row++)
    for (col = 0; col < width; col++)
    {
      if (col == (unsigned)border && row >= (unsigned)border && row < (unsigned)(height - border))
        col = width - border;
      memset(sum, 0, sizeof sum);
      for (y = row - 1; y != row + 2; y++)
        for (x = col - 1; x != col + 2; x++)
          if (y < height && x < width)
          {
            f = fcol(y, x, filters);
            sum[f] += image[y * width + x][f];
            sum[f + 4]++;
          }
      f = fcol(row, col, filters);
      FORCC if (c != f && sum[c + 4]) image[row * width + col][c] =
          sum[c] / sum[c + 4];
    }
}

static void lin_interpolate_loop(int *code, int size, ushort (*image)[4], int width, int height, unsigned filters)
{
  int row;
  for (row = 1; row < height - 1; row++)
  {
    int col, *ip;
    ushort *pix;
    for (col = 1; col < width - 1; col++)
    {
      int i;
      int sum[4];
      pix = image[row * width + col];
      ip = code + ((((row % size) * 16) + (col % size)) * 32);
      memset(sum, 0, sizeof sum);
      for (i = *ip++; i--; ip += 3)
        sum[ip[2]] += pix[ip[0]] << ip[1];
      for (i = 3; --i; ip += 2)
        pix[ip[0]] = sum[ip[0]] * ip[1] >> 8;
    }
  }
}

static void lin_interpolate(ushort (*image)[4], int width, int height, unsigned filters)
{
  int *code = (int *)calloc(16 * 16 * 32, sizeof(int));
  int size = 16, *ip, sum[4];
  int f, c, x, y, row, col, shift, color;

  if (filters == 9)
    size = 6;
  border_interpolate(1, image, width, height, filters);
  for (row = 0; row < size; row++)
    for (col = 0; col < size; col++)
    {
      ip = code + (((row * 16) + col) * 32) + 1;
      f = fcol(row, col, filters);
      memset(sum, 0, sizeof sum);
      for (y = -1; y <= 1; y++)
        for (x = -1; x <= 1; x++)
        {
          shift = (y == 0) + (x == 0);
          color = fcol(row + y + 48, col + x + 48, filters);
          if (color == f)
            continue;
          *ip++ = (width * y + x) * 4 + color;
          *ip++ = shift;
          *ip++ = color;
          sum[color] += 1 << shift;
        }
      code[(row * 16 + col) * 32] = (int)((ip - (code + ((row * 16) + col) * 32)) / 3);
      FORCC
      if (c != f)
      {
        *ip++ = c;
        *ip++ = sum[c] > 0 ? 256 / sum[c] : 0;
      }
    }
  lin_interpolate_loop(code, size, image, width, height, filters);
  free(code);
}

static void vng_interpolate(ushort (*image)[4], int width, int height, unsigned filters)
{
  const signed char *cp;  /* FIXED: Made local instead of static for thread-safety */
  static const signed char
      terms[] =
          {-2, -2, +0,   -1, 0,  0x01, -2, -2, +0,   +0, 1,  0x01, -2, -1, -1,
           +0, 0,  0x01, -2, -1, +0,   -1, 0,  0x02, -2, -1, +0,   +0, 0,  0x03,
           -2, -1, +0,   +1, 1,  0x01, -2, +0, +0,   -1, 0,  0x06, -2, +0, +0,
           +0, 1,  0x02, -2, +0, +0,   +1, 0,  0x03, -2, +1, -1,   +0, 0,  0x04,
           -2, +1, +0,   -1, 1,  0x04, -2, +1, +0,   +0, 0,  0x06, -2, +1, +0,
           +1, 0,  0x02, -2, +2, +0,   +0, 1,  0x04, -2, +2, +0,   +1, 0,  0x04,
           -1, -2, -1,   +0, 0,  -128, -1, -2, +0,   -1, 0,  0x01, -1, -2, +1,
           -1, 0,  0x01, -1, -2, +1,   +0, 1,  0x01, -1, -1, -1,   +1, 0,  -120,
           -1, -1, +1,   -2, 0,  0x40, -1, -1, +1,   -1, 0,  0x22, -1, -1, +1,
           +0, 0,  0x33, -1, -1, +1,   +1, 1,  0x11, -1, +0, -1,   +2, 0,  0x08,
           -1, +0, +0,   -1, 0,  0x44, -1, +0, +0,   +1, 0,  0x11, -1, +0, +1,
           -2, 1,  0x40, -1, +0, +1,   -1, 0,  0x66, -1, +0, +1,   +0, 1,  0x22,
           -1, +0, +1,   +1, 0,  0x33, -1, +0, +1,   +2, 1,  0x10, -1, +1, +1,
           -1, 1,  0x44, -1, +1, +1,   +0, 0,  0x66, -1, +1, +1,   +1, 0,  0x22,
           -1, +1, +1,   +2, 0,  0x10, -1, +2, +0,   +1, 0,  0x04, -1, +2, +1,
           +0, 1,  0x04, -1, +2, +1,   +1, 0,  0x04, +0, -2, +0,   +0, 1,  -128,
           +0, -1, +0,   +1, 1,  -120, +0, -1, +1,   -2, 0,  0x40, +0, -1, +1,
           +0, 0,  0x11, +0, -1, +2,   -2, 0,  0x40, +0, -1, +2,   -1, 0,  0x20,
           +0, -1, +2,   +0, 0,  0x30, +0, -1, +2,   +1, 1,  0x10, +0, +0, +0,
           +2, 1,  0x08, +0, +0, +2,   -2, 1,  0x40, +0, +0, +2,   -1, 0,  0x60,
           +0, +0, +2,   +0, 1,  0x20, +0, +0, +2,   +1, 0,  0x30, +0, +0, +2,
           +2, 1,  0x10, +0, +1, +1,   +0, 0,  0x44, +0, +1, +1,   +2, 0,  0x10,
           +0, +1, +2,   -1, 1,  0x40, +0, +1, +2,   +0, 0,  0x60, +0, +1, +2,
           +1, 0,  0x20, +0, +1, +2,   +2, 0,  0x10, +1, -2, +1,   +0, 0,  -128,
           +1, -1, +1,   +1, 0,  -120, +1, +0, +1,   +2, 0,  0x08, +1, +0, +2,
           -1, 0,  0x40, +1, +0, +2,   +1, 0,  0x10},
      chood[] = {-1, -1, -1, 0, -1, +1, 0, +1, +1, +1, +1, 0, +1, -1, 0, -1};
  ushort(*brow[5])[4], *pix;
  int prow = 8, pcol = 2, *ip, *code[16][16], gval[8], gmin, gmax, sum[4];
  int row, col, x, y, x1, x2, y1, y2, t, weight, grads, color, diag;
  int g, diff, thold, num, c;

  lin_interpolate(image, width, height, filters);

  if (filters == 1)
    prow = pcol = 16;
  if (filters == 9)
    prow = pcol = 6;
  ip = (int *)calloc(prow * pcol, 1280);
  for (row = 0; row < prow; row++) /* Precalculate for VNG */
    for (col = 0; col < pcol; col++)
    {
      code[row][col] = ip;
      for (cp = terms, t = 0; t < 64; t++)
      {
        y1 = *cp++;
        x1 = *cp++;
        y2 = *cp++;
        x2 = *cp++;
        weight = *cp++;
        grads = *cp++;
        color = fcol(row + y1 + 144, col + x1 + 144, filters);
        if (fcol(row + y2 + 144, col + x2 + 144, filters) != color)
          continue;
        diag = (fcol(row, col + 1, filters) == color && fcol(row + 1, col, filters) == color) ? 2
                                                                            : 1;
        if (ABS(y1 - y2) == diag && ABS(x1 - x2) == diag)
          continue;
        *ip++ = (y1 * width + x1) * 4 + color;
        *ip++ = (y2 * width + x2) * 4 + color;
        *ip++ = weight;
        for (g = 0; g < 8; g++)
          if (grads & 1 << g)
            *ip++ = g;
        *ip++ = -1;
      }
      *ip++ = INT_MAX;
      for (cp = chood, g = 0; g < 8; g++)
      {
        y = *cp++;
        x = *cp++;
        *ip++ = (y * width + x) * 4;
        color = fcol(row, col, filters);
        if (fcol(row + y + 144, col + x + 144, filters) != color &&
            fcol(row + y * 2 + 144, col + x * 2 + 144, filters) == color)
          *ip++ = (y * width + x) * 8 + color;
        else
          *ip++ = 0;
      }
    }
  brow[4] = (ushort(*)[4])calloc(width * 3, sizeof **brow);
  for (row = 0; row < 3; row++)
    brow[row] = brow[4] + row * width;
  for (row = 2; row < height - 2; row++)
  { /* Do VNG interpolation */
    for (col = 2; col < width - 2; col++)
    {
      pix = image[row * width + col];
      ip = code[row % prow][col % pcol];
      memset(gval, 0, sizeof gval);
      color = fcol(row, col, filters);
      while ((g = ip[0]) != INT_MAX)
      { /* Calculate gradients */
        diff = ABS(pix[g] - pix[ip[1]]) << ip[2];
        gval[ip[3]] += diff;
        ip += 5;
        if ((g = ip[-1]) == -1)
          continue;
        gval[g] += diff;
        while ((g = *ip++) != -1)
          gval[g] += diff;
      }
      ip++;
      gmin = gmax = gval[0]; /* Choose a threshold */
      for (g = 1; g < 8; g++)
      {
        if (gmin > gval[g])
          gmin = gval[g];
        if (gmax < gval[g])
          gmax = gval[g];
      }
      if (gmax == 0)
      {
        memcpy(brow[2][col], pix, sizeof *image);
        continue;
      }
      thold = gmin + (gmax >> 1);
      memset(sum, 0, sizeof sum);
      for (num = g = 0; g < 8; g++, ip += 2)
      { /* Average the neighbors */
        if (gval[g] <= thold)
        {
          FORCC
          if (c == color && ip[1])
            sum[c] += (pix[c] + pix[ip[1]]) >> 1;
          else
            sum[c] += pix[ip[0] + c];
          num++;
        }
      }
      FORCC
      { /* Save to buffer */
        t = pix[color];
        if (c != color)
          t += (sum[c] - sum[color]) / num;
        brow[2][col][c] = CLIP(t);
      }
    }
    if (row > 3) /* Write buffer to image */
      memcpy(image[(row - 2) * width + 2], brow[0] + 2,
             (width - 4) * sizeof *image);
    for (g = 0; g < 4; g++)
      brow[(g - 1) & 3] = brow[g];
  }
  memcpy(image[(row - 2) * width + 2], brow[0] + 2,
         (width - 4) * sizeof *image);
  memcpy(image[(row - 1) * width + 2], brow[1] + 2,
         (width - 4) * sizeof *image);
  free(brow[4]);
  free(code[0][0]);
}

/* Python wrapper function */
static PyObject* py_vng_demosaic(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyArrayObject *cfa_array;
    PyArrayObject *output_array;
    const char *pattern_str;
    ushort (*image)[4] = NULL;
    int width, height;
    unsigned filters;
    
    static char *kwlist[] = {"cfa", "pattern", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!s", kwlist,
                                     &PyArray_Type, &cfa_array,
                                     &pattern_str)) {
        return NULL;
    }
    
    /* Validate input array */
    if (PyArray_NDIM(cfa_array) != 2 || PyArray_TYPE(cfa_array) != NPY_UINT16) {
        PyErr_SetString(PyExc_ValueError, "CFA must be 2D uint16 array");
        return NULL;
    }
    
    /* Get dimensions */
    height = (int)PyArray_DIM(cfa_array, 0);
    width = (int)PyArray_DIM(cfa_array, 1);
    
    /* Parse Bayer pattern */
    if (strcmp(pattern_str, "RGGB") == 0) {
        filters = 0x94949494;
    } else if (strcmp(pattern_str, "BGGR") == 0) {
        filters = 0x16161616;
    } else if (strcmp(pattern_str, "GRBG") == 0) {
        filters = 0x61616161;
    } else if (strcmp(pattern_str, "GBRG") == 0) {
        filters = 0x49494949;
    } else {
        PyErr_SetString(PyExc_ValueError, "Invalid pattern. Must be RGGB, BGGR, GRBG, or GBRG");
        return NULL;
    }
    
    /* Allocate image buffer with 4 channels (RGBG2) */
    image = (ushort(*)[4])calloc(width * height, sizeof(*image));
    if (!image) {
        PyErr_NoMemory();
        return NULL;
    }
    
    /* Copy CFA data to image buffer (only the appropriate channel) */
    ushort *cfa_data = (ushort *)PyArray_DATA(cfa_array);
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            int c = fcol(row, col, filters);
            image[row * width + col][c] = cfa_data[row * width + col];
        }
    }
    
    /* Run VNG interpolation */
    Py_BEGIN_ALLOW_THREADS
    vng_interpolate(image, width, height, filters);
    Py_END_ALLOW_THREADS
    
    /* Create output array (height, width, 3) */
    npy_intp dims[3] = {height, width, 3};
    output_array = (PyArrayObject *)PyArray_SimpleNew(3, dims, NPY_UINT16);
    if (!output_array) {
        free(image);
        return NULL;
    }
    
    /* Copy RGB data to output (skip G2 channel) */
    ushort *out_data = (ushort *)PyArray_DATA(output_array);
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            int idx = (row * width + col) * 3;
            out_data[idx + 0] = image[row * width + col][0];  /* R */
            out_data[idx + 1] = image[row * width + col][1];  /* G */
            out_data[idx + 2] = image[row * width + col][2];  /* B */
        }
    }
    
    free(image);
    
    return (PyObject *)output_array;
}

/* Module methods */
static PyMethodDef VngMethods[] = {
    {"vng_demosaic", (PyCFunction)py_vng_demosaic, METH_VARARGS | METH_KEYWORDS,
     "VNG demosaicing algorithm\n\n"
     "Args:\n"
     "    cfa (np.ndarray): 2D uint16 CFA array\n"
     "    pattern (str): Bayer pattern ('RGGB', 'BGGR', 'GRBG', or 'GBRG')\n\n"
     "Returns:\n"
     "    np.ndarray: RGB image (height, width, 3) uint16\n"},
    {NULL, NULL, 0, NULL}
};

/* Module definition */
static struct PyModuleDef vngmodule = {
    PyModuleDef_HEAD_INIT,
    "_vng",
    "VNG demosaicing C extension",
    -1,
    VngMethods
};

/* Module initialization */
PyMODINIT_FUNC PyInit__vng(void)
{
    import_array();
    return PyModule_Create(&vngmodule);
}
