/*
 * DNG SDK Color Processing - Standalone Python wrapper
 * 
 * This module provides Python bindings for DNG SDK color processing
 * algorithms, implemented as standalone code extracted from the SDK:
 * - Color temperature/tint to white point conversion
 * - Dual/triple illuminant matrix interpolation
 * - CameraCalibration, AnalogBalance matrices
 * - ForwardMatrix support
 * - ProfileHueSatMap (3D HSV LUT)
 * - ProfileLookTable (second HSV pass)
 * - ProfileToneCurve (custom or ACR3 default)
 * - Color matrix transforms
 * 
 * Based on Adobe DNG SDK 1.7.1
 * Original Copyright 2006-2024 Adobe Systems Incorporated
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <cmath>
#include <cstring>
#include <algorithm>
#include <vector>

//=============================================================================
// Color Temperature Conversion (from dng_temperature.cpp)
// Table from Wyszecki & Stiles, "Color Science", second edition, page 228
//=============================================================================

static const double kTintScale = -3000.0;

struct ruvt {
    double r;
    double u;
    double v;
    double t;
};

static const ruvt kTempTable[] = {
    {   0, 0.18006, 0.26352, -0.24341 },
    {  10, 0.18066, 0.26589, -0.25479 },
    {  20, 0.18133, 0.26846, -0.26876 },
    {  30, 0.18208, 0.27119, -0.28539 },
    {  40, 0.18293, 0.27407, -0.30470 },
    {  50, 0.18388, 0.27709, -0.32675 },
    {  60, 0.18494, 0.28021, -0.35156 },
    {  70, 0.18611, 0.28342, -0.37915 },
    {  80, 0.18740, 0.28668, -0.40955 },
    {  90, 0.18880, 0.28997, -0.44278 },
    { 100, 0.19032, 0.29326, -0.47888 },
    { 125, 0.19462, 0.30141, -0.58204 },
    { 150, 0.19962, 0.30921, -0.70471 },
    { 175, 0.20525, 0.31647, -0.84901 },
    { 200, 0.21142, 0.32312, -1.0182  },
    { 225, 0.21807, 0.32909, -1.2168  },
    { 250, 0.22511, 0.33439, -1.4512  },
    { 275, 0.23247, 0.33904, -1.7298  },
    { 300, 0.24010, 0.34308, -2.0637  },
    { 325, 0.24702, 0.34655, -2.4681  },
    { 350, 0.25591, 0.34951, -2.9641  },
    { 375, 0.26400, 0.35200, -3.5814  },
    { 400, 0.27218, 0.35407, -4.3633  },
    { 425, 0.28039, 0.35577, -5.3762  },
    { 450, 0.28863, 0.35714, -6.7262  },
    { 475, 0.29685, 0.35823, -8.5955  },
    { 500, 0.30505, 0.35907, -11.324  },
    { 525, 0.31320, 0.35968, -15.628  },
    { 550, 0.32129, 0.36011, -23.325  },
    { 575, 0.32931, 0.36038, -40.770  },
    { 600, 0.33724, 0.36051, -116.45  }
};

static void xy_to_temp_tint(double x, double y, double* temperature, double* tint) {
    // Convert to uv space
    double u = 2.0 * x / (1.5 - x + 6.0 * y);
    double v = 3.0 * y / (1.5 - x + 6.0 * y);
    
    double last_dt = 0.0;
    double last_dv = 0.0;
    double last_du = 0.0;
    
    for (int index = 1; index <= 30; index++) {
        double du = 1.0;
        double dv = kTempTable[index].t;
        
        double len = sqrt(1.0 + dv * dv);
        du /= len;
        dv /= len;
        
        double uu = u - kTempTable[index].u;
        double vv = v - kTempTable[index].v;
        
        double dt = -uu * dv + vv * du;
        
        if (dt <= 0.0 || index == 30) {
            if (dt > 0.0) dt = 0.0;
            dt = -dt;
            
            double f;
            if (index == 1) {
                f = 0.0;
            } else {
                f = dt / (last_dt + dt);
            }
            
            *temperature = 1.0E6 / (kTempTable[index - 1].r * f +
                                    kTempTable[index].r * (1.0 - f));
            
            uu = u - (kTempTable[index - 1].u * f +
                      kTempTable[index].u * (1.0 - f));
            vv = v - (kTempTable[index - 1].v * f +
                      kTempTable[index].v * (1.0 - f));
            
            du = du * (1.0 - f) + last_du * f;
            dv = dv * (1.0 - f) + last_dv * f;
            
            len = sqrt(du * du + dv * dv);
            du /= len;
            dv /= len;
            
            *tint = (uu * du + vv * dv) * kTintScale;
            break;
        }
        
        last_dt = dt;
        last_du = du;
        last_dv = dv;
    }
}

static void temp_tint_to_xy(double temperature, double tint, double* x, double* y) {
    // Find inverse temperature
    double r = 1.0E6 / temperature;
    
    // Find bounding table entries
    int index;
    for (index = 30; index > 0; index--) {
        if (r >= kTempTable[index].r) break;
    }
    
    // Interpolate
    double f = (r - kTempTable[index].r) / 
               (kTempTable[index + 1].r - kTempTable[index].r);
    
    double u = kTempTable[index].u * (1.0 - f) + kTempTable[index + 1].u * f;
    double v = kTempTable[index].v * (1.0 - f) + kTempTable[index + 1].v * f;
    
    // Interpolate slope
    double du = 1.0;
    double dv0 = kTempTable[index].t;
    double dv1 = kTempTable[index + 1].t;
    
    double len0 = sqrt(1.0 + dv0 * dv0);
    double len1 = sqrt(1.0 + dv1 * dv1);
    
    double du0 = 1.0 / len0;
    double dv0n = dv0 / len0;
    double du1 = 1.0 / len1;
    double dv1n = dv1 / len1;
    
    du = du0 * (1.0 - f) + du1 * f;
    double dv = dv0n * (1.0 - f) + dv1n * f;
    
    double len = sqrt(du * du + dv * dv);
    du /= len;
    dv /= len;
    
    // Apply tint offset
    double offset = tint / kTintScale;
    u += offset * du;
    v += offset * dv;
    
    // Convert uv to xy
    *x = 1.5 * u / (u - 4.0 * v + 2.0);
    *y = v / (u - 4.0 * v + 2.0);
}

//=============================================================================
// ACR3 Default Tone Curve (from dng_render.cpp)
//=============================================================================

// ACR3 tone curve table from dng_render.cpp dng_tone_curve_acr3_default::Evaluate()
// 257 entries, input [0,1] maps to output [0,1]
static const float kACR3ToneCurve[] = {
    0.00000f, 0.00078f, 0.00160f, 0.00242f,
    0.00314f, 0.00385f, 0.00460f, 0.00539f,
    0.00623f, 0.00712f, 0.00806f, 0.00906f,
    0.01012f, 0.01122f, 0.01238f, 0.01359f,
    0.01485f, 0.01616f, 0.01751f, 0.01890f,
    0.02033f, 0.02180f, 0.02331f, 0.02485f,
    0.02643f, 0.02804f, 0.02967f, 0.03134f,
    0.03303f, 0.03475f, 0.03648f, 0.03824f,
    0.04002f, 0.04181f, 0.04362f, 0.04545f,
    0.04730f, 0.04916f, 0.05103f, 0.05292f,
    0.05483f, 0.05675f, 0.05868f, 0.06063f,
    0.06259f, 0.06457f, 0.06655f, 0.06856f,
    0.07057f, 0.07259f, 0.07463f, 0.07668f,
    0.07874f, 0.08081f, 0.08290f, 0.08499f,
    0.08710f, 0.08921f, 0.09134f, 0.09348f,
    0.09563f, 0.09779f, 0.09996f, 0.10214f,
    0.10433f, 0.10652f, 0.10873f, 0.11095f,
    0.11318f, 0.11541f, 0.11766f, 0.11991f,
    0.12218f, 0.12445f, 0.12673f, 0.12902f,
    0.13132f, 0.13363f, 0.13595f, 0.13827f,
    0.14061f, 0.14295f, 0.14530f, 0.14765f,
    0.15002f, 0.15239f, 0.15477f, 0.15716f,
    0.15956f, 0.16197f, 0.16438f, 0.16680f,
    0.16923f, 0.17166f, 0.17410f, 0.17655f,
    0.17901f, 0.18148f, 0.18395f, 0.18643f,
    0.18891f, 0.19141f, 0.19391f, 0.19641f,
    0.19893f, 0.20145f, 0.20398f, 0.20651f,
    0.20905f, 0.21160f, 0.21416f, 0.21672f,
    0.21929f, 0.22185f, 0.22440f, 0.22696f,
    0.22950f, 0.23204f, 0.23458f, 0.23711f,
    0.23963f, 0.24215f, 0.24466f, 0.24717f,
    0.24967f, 0.25216f, 0.25465f, 0.25713f,
    0.25961f, 0.26208f, 0.26454f, 0.26700f,
    0.26945f, 0.27189f, 0.27433f, 0.27676f,
    0.27918f, 0.28160f, 0.28401f, 0.28641f,
    0.28881f, 0.29120f, 0.29358f, 0.29596f,
    0.29833f, 0.30069f, 0.30305f, 0.30540f,
    0.30774f, 0.31008f, 0.31241f, 0.31473f,
    0.31704f, 0.31935f, 0.32165f, 0.32395f,
    0.32623f, 0.32851f, 0.33079f, 0.33305f,
    0.33531f, 0.33756f, 0.33981f, 0.34205f,
    0.34428f, 0.34650f, 0.34872f, 0.35093f,
    0.35313f, 0.35532f, 0.35751f, 0.35969f,
    0.36187f, 0.36404f, 0.36620f, 0.36835f,
    0.37050f, 0.37264f, 0.37477f, 0.37689f,
    0.37901f, 0.38112f, 0.38323f, 0.38533f,
    0.38742f, 0.38950f, 0.39158f, 0.39365f,
    0.39571f, 0.39777f, 0.39982f, 0.40186f,
    0.40389f, 0.40592f, 0.40794f, 0.40996f,
    0.41197f, 0.41397f, 0.41596f, 0.41795f,
    0.41993f, 0.42191f, 0.42388f, 0.42584f,
    0.42779f, 0.42974f, 0.43168f, 0.43362f,
    0.43554f, 0.43747f, 0.43938f, 0.44129f,
    0.44319f, 0.44509f, 0.44698f, 0.44886f,
    0.45073f, 0.45260f, 0.45447f, 0.45632f,
    0.45817f, 0.46002f, 0.46186f, 0.46369f,
    0.46551f, 0.46733f, 0.46914f, 0.47095f,
    0.47275f, 0.47454f, 0.47633f, 0.47811f,
    0.47989f, 0.48166f, 0.48342f, 0.48518f,
    0.48693f, 0.48867f, 0.49041f, 0.49214f,
    0.49387f, 0.49559f, 0.49730f, 0.49901f,
    0.50072f, 0.50241f, 0.50410f, 0.50579f,
    0.50747f, 0.50914f, 0.51081f, 0.51247f,
    0.51413f, 0.51578f, 0.51742f, 0.51906f,
    0.52069f, 0.52232f, 0.52394f, 0.52556f,
    0.52717f, 0.52878f, 0.53038f, 0.53197f,
    0.53356f, 0.53514f, 0.53672f, 0.53829f,
    0.53986f, 0.54142f, 0.54297f, 0.54452f,
    0.54607f, 0.54761f, 0.54914f, 0.55067f,
    0.55220f, 0.55371f, 0.55523f, 0.55673f,
    0.55824f, 0.55973f, 0.56123f, 0.56271f,
    0.56420f, 0.56567f, 0.56715f, 0.56861f,
    0.57007f, 0.57153f, 0.57298f, 0.57443f,
    0.57587f, 0.57731f, 0.57874f, 0.58017f,
    0.58159f, 0.58301f, 0.58443f, 0.58583f,
    0.58724f, 0.58864f, 0.59003f, 0.59142f,
    0.59281f, 0.59419f, 0.59556f, 0.59694f,
    0.59830f, 0.59966f, 0.60102f, 0.60238f,
    0.60373f, 0.60507f, 0.60641f, 0.60775f,
    0.60908f, 0.61040f, 0.61173f, 0.61305f,
    0.61436f, 0.61567f, 0.61698f, 0.61828f,
    0.61957f, 0.62087f, 0.62216f, 0.62344f,
    0.62472f, 0.62600f, 0.62727f, 0.62854f,
    0.62980f, 0.63106f, 0.63232f, 0.63357f,
    0.63482f, 0.63606f, 0.63730f, 0.63854f,
    0.63977f, 0.64100f, 0.64222f, 0.64344f,
    0.64466f, 0.64587f, 0.64708f, 0.64829f,
    0.64949f, 0.65069f, 0.65188f, 0.65307f,
    0.65426f, 0.65544f, 0.65662f, 0.65779f,
    0.65897f, 0.66013f, 0.66130f, 0.66246f,
    0.66362f, 0.66477f, 0.66592f, 0.66707f,
    0.66821f, 0.66935f, 0.67048f, 0.67162f,
    0.67275f, 0.67387f, 0.67499f, 0.67611f,
    0.67723f, 0.67834f, 0.67945f, 0.68055f,
    0.68165f, 0.68275f, 0.68385f, 0.68494f,
    0.68603f, 0.68711f, 0.68819f, 0.68927f,
    0.69035f, 0.69142f, 0.69249f, 0.69355f,
    0.69461f, 0.69567f, 0.69673f, 0.69778f,
    0.69883f, 0.69988f, 0.70092f, 0.70196f,
    0.70300f, 0.70403f, 0.70506f, 0.70609f,
    0.70711f, 0.70813f, 0.70915f, 0.71017f,
    0.71118f, 0.71219f, 0.71319f, 0.71420f,
    0.71520f, 0.71620f, 0.71719f, 0.71818f,
    0.71917f, 0.72016f, 0.72114f, 0.72212f,
    0.72309f, 0.72407f, 0.72504f, 0.72601f,
    0.72697f, 0.72794f, 0.72890f, 0.72985f,
    0.73081f, 0.73176f, 0.73271f, 0.73365f,
    0.73460f, 0.73554f, 0.73647f, 0.73741f,
    0.73834f, 0.73927f, 0.74020f, 0.74112f,
    0.74204f, 0.74296f, 0.74388f, 0.74479f,
    0.74570f, 0.74661f, 0.74751f, 0.74842f,
    0.74932f, 0.75021f, 0.75111f, 0.75200f,
    0.75289f, 0.75378f, 0.75466f, 0.75555f,
    0.75643f, 0.75730f, 0.75818f, 0.75905f,
    0.75992f, 0.76079f, 0.76165f, 0.76251f,
    0.76337f, 0.76423f, 0.76508f, 0.76594f,
    0.76679f, 0.76763f, 0.76848f, 0.76932f,
    0.77016f, 0.77100f, 0.77183f, 0.77267f,
    0.77350f, 0.77432f, 0.77515f, 0.77597f,
    0.77680f, 0.77761f, 0.77843f, 0.77924f,
    0.78006f, 0.78087f, 0.78167f, 0.78248f,
    0.78328f, 0.78408f, 0.78488f, 0.78568f,
    0.78647f, 0.78726f, 0.78805f, 0.78884f,
    0.78962f, 0.79040f, 0.79118f, 0.79196f,
    0.79274f, 0.79351f, 0.79428f, 0.79505f,
    0.79582f, 0.79658f, 0.79735f, 0.79811f,
    0.79887f, 0.79962f, 0.80038f, 0.80113f,
    0.80188f, 0.80263f, 0.80337f, 0.80412f,
    0.80486f, 0.80560f, 0.80634f, 0.80707f,
    0.80780f, 0.80854f, 0.80926f, 0.80999f,
    0.81072f, 0.81144f, 0.81216f, 0.81288f,
    0.81360f, 0.81431f, 0.81503f, 0.81574f,
    0.81645f, 0.81715f, 0.81786f, 0.81856f,
    0.81926f, 0.81996f, 0.82066f, 0.82135f,
    0.82205f, 0.82274f, 0.82343f, 0.82412f,
    0.82480f, 0.82549f, 0.82617f, 0.82685f,
    0.82753f, 0.82820f, 0.82888f, 0.82955f,
    0.83022f, 0.83089f, 0.83155f, 0.83222f,
    0.83288f, 0.83354f, 0.83420f, 0.83486f,
    0.83552f, 0.83617f, 0.83682f, 0.83747f,
    0.83812f, 0.83877f, 0.83941f, 0.84005f,
    0.84069f, 0.84133f, 0.84197f, 0.84261f,
    0.84324f, 0.84387f, 0.84450f, 0.84513f,
    0.84576f, 0.84639f, 0.84701f, 0.84763f,
    0.84825f, 0.84887f, 0.84949f, 0.85010f,
    0.85071f, 0.85132f, 0.85193f, 0.85254f,
    0.85315f, 0.85375f, 0.85436f, 0.85496f,
    0.85556f, 0.85615f, 0.85675f, 0.85735f,
    0.85794f, 0.85853f, 0.85912f, 0.85971f,
    0.86029f, 0.86088f, 0.86146f, 0.86204f,
    0.86262f, 0.86320f, 0.86378f, 0.86435f,
    0.86493f, 0.86550f, 0.86607f, 0.86664f,
    0.86720f, 0.86777f, 0.86833f, 0.86889f,
    0.86945f, 0.87001f, 0.87057f, 0.87113f,
    0.87168f, 0.87223f, 0.87278f, 0.87333f,
    0.87388f, 0.87443f, 0.87497f, 0.87552f,
    0.87606f, 0.87660f, 0.87714f, 0.87768f,
    0.87821f, 0.87875f, 0.87928f, 0.87981f,
    0.88034f, 0.88087f, 0.88140f, 0.88192f,
    0.88244f, 0.88297f, 0.88349f, 0.88401f,
    0.88453f, 0.88504f, 0.88556f, 0.88607f,
    0.88658f, 0.88709f, 0.88760f, 0.88811f,
    0.88862f, 0.88912f, 0.88963f, 0.89013f,
    0.89063f, 0.89113f, 0.89163f, 0.89212f,
    0.89262f, 0.89311f, 0.89360f, 0.89409f,
    0.89458f, 0.89507f, 0.89556f, 0.89604f,
    0.89653f, 0.89701f, 0.89749f, 0.89797f,
    0.89845f, 0.89892f, 0.89940f, 0.89987f,
    0.90035f, 0.90082f, 0.90129f, 0.90176f,
    0.90222f, 0.90269f, 0.90316f, 0.90362f,
    0.90408f, 0.90454f, 0.90500f, 0.90546f,
    0.90592f, 0.90637f, 0.90683f, 0.90728f,
    0.90773f, 0.90818f, 0.90863f, 0.90908f,
    0.90952f, 0.90997f, 0.91041f, 0.91085f,
    0.91130f, 0.91173f, 0.91217f, 0.91261f,
    0.91305f, 0.91348f, 0.91392f, 0.91435f,
    0.91478f, 0.91521f, 0.91564f, 0.91606f,
    0.91649f, 0.91691f, 0.91734f, 0.91776f,
    0.91818f, 0.91860f, 0.91902f, 0.91944f,
    0.91985f, 0.92027f, 0.92068f, 0.92109f,
    0.92150f, 0.92191f, 0.92232f, 0.92273f,
    0.92314f, 0.92354f, 0.92395f, 0.92435f,
    0.92475f, 0.92515f, 0.92555f, 0.92595f,
    0.92634f, 0.92674f, 0.92713f, 0.92753f,
    0.92792f, 0.92831f, 0.92870f, 0.92909f,
    0.92947f, 0.92986f, 0.93025f, 0.93063f,
    0.93101f, 0.93139f, 0.93177f, 0.93215f,
    0.93253f, 0.93291f, 0.93328f, 0.93366f,
    0.93403f, 0.93440f, 0.93478f, 0.93515f,
    0.93551f, 0.93588f, 0.93625f, 0.93661f,
    0.93698f, 0.93734f, 0.93770f, 0.93807f,
    0.93843f, 0.93878f, 0.93914f, 0.93950f,
    0.93986f, 0.94021f, 0.94056f, 0.94092f,
    0.94127f, 0.94162f, 0.94197f, 0.94231f,
    0.94266f, 0.94301f, 0.94335f, 0.94369f,
    0.94404f, 0.94438f, 0.94472f, 0.94506f,
    0.94540f, 0.94573f, 0.94607f, 0.94641f,
    0.94674f, 0.94707f, 0.94740f, 0.94774f,
    0.94807f, 0.94839f, 0.94872f, 0.94905f,
    0.94937f, 0.94970f, 0.95002f, 0.95035f,
    0.95067f, 0.95099f, 0.95131f, 0.95163f,
    0.95194f, 0.95226f, 0.95257f, 0.95289f,
    0.95320f, 0.95351f, 0.95383f, 0.95414f,
    0.95445f, 0.95475f, 0.95506f, 0.95537f,
    0.95567f, 0.95598f, 0.95628f, 0.95658f,
    0.95688f, 0.95718f, 0.95748f, 0.95778f,
    0.95808f, 0.95838f, 0.95867f, 0.95897f,
    0.95926f, 0.95955f, 0.95984f, 0.96013f,
    0.96042f, 0.96071f, 0.96100f, 0.96129f,
    0.96157f, 0.96186f, 0.96214f, 0.96242f,
    0.96271f, 0.96299f, 0.96327f, 0.96355f,
    0.96382f, 0.96410f, 0.96438f, 0.96465f,
    0.96493f, 0.96520f, 0.96547f, 0.96574f,
    0.96602f, 0.96629f, 0.96655f, 0.96682f,
    0.96709f, 0.96735f, 0.96762f, 0.96788f,
    0.96815f, 0.96841f, 0.96867f, 0.96893f,
    0.96919f, 0.96945f, 0.96971f, 0.96996f,
    0.97022f, 0.97047f, 0.97073f, 0.97098f,
    0.97123f, 0.97149f, 0.97174f, 0.97199f,
    0.97223f, 0.97248f, 0.97273f, 0.97297f,
    0.97322f, 0.97346f, 0.97371f, 0.97395f,
    0.97419f, 0.97443f, 0.97467f, 0.97491f,
    0.97515f, 0.97539f, 0.97562f, 0.97586f,
    0.97609f, 0.97633f, 0.97656f, 0.97679f,
    0.97702f, 0.97725f, 0.97748f, 0.97771f,
    0.97794f, 0.97817f, 0.97839f, 0.97862f,
    0.97884f, 0.97907f, 0.97929f, 0.97951f,
    0.97973f, 0.97995f, 0.98017f, 0.98039f,
    0.98061f, 0.98082f, 0.98104f, 0.98125f,
    0.98147f, 0.98168f, 0.98189f, 0.98211f,
    0.98232f, 0.98253f, 0.98274f, 0.98295f,
    0.98315f, 0.98336f, 0.98357f, 0.98377f,
    0.98398f, 0.98418f, 0.98438f, 0.98458f,
    0.98478f, 0.98498f, 0.98518f, 0.98538f,
    0.98558f, 0.98578f, 0.98597f, 0.98617f,
    0.98636f, 0.98656f, 0.98675f, 0.98694f,
    0.98714f, 0.98733f, 0.98752f, 0.98771f,
    0.98789f, 0.98808f, 0.98827f, 0.98845f,
    0.98864f, 0.98882f, 0.98901f, 0.98919f,
    0.98937f, 0.98955f, 0.98973f, 0.98991f,
    0.99009f, 0.99027f, 0.99045f, 0.99063f,
    0.99080f, 0.99098f, 0.99115f, 0.99133f,
    0.99150f, 0.99167f, 0.99184f, 0.99201f,
    0.99218f, 0.99235f, 0.99252f, 0.99269f,
    0.99285f, 0.99302f, 0.99319f, 0.99335f,
    0.99351f, 0.99368f, 0.99384f, 0.99400f,
    0.99416f, 0.99432f, 0.99448f, 0.99464f,
    0.99480f, 0.99495f, 0.99511f, 0.99527f,
    0.99542f, 0.99558f, 0.99573f, 0.99588f,
    0.99603f, 0.99619f, 0.99634f, 0.99649f,
    0.99664f, 0.99678f, 0.99693f, 0.99708f,
    0.99722f, 0.99737f, 0.99751f, 0.99766f,
    0.99780f, 0.99794f, 0.99809f, 0.99823f,
    0.99837f, 0.99851f, 0.99865f, 0.99879f,
    0.99892f, 0.99906f, 0.99920f, 0.99933f,
    0.99947f, 0.99960f, 0.99974f, 0.99987f,
    1.00000f
};

static const int kACR3TableSize = sizeof(kACR3ToneCurve) / sizeof(kACR3ToneCurve[0]);

static float evaluate_acr3_curve(float x) {
    if (x <= 0.0f) return 0.0f;
    if (x >= 1.0f) return 1.0f;
    
    float idx = x * (float)(kACR3TableSize - 1);
    int i = (int)idx;
    if (i >= kACR3TableSize - 1) i = kACR3TableSize - 2;
    
    float fract = idx - (float)i;
    return kACR3ToneCurve[i] * (1.0f - fract) + kACR3ToneCurve[i + 1] * fract;
}

//=============================================================================
// RGB <-> HSV Conversion (from dng_utils.h)
// H range: 0-6, S/V range: 0-1
//=============================================================================

static inline void rgb_to_hsv(float r, float g, float b, float& h, float& s, float& v) {
    v = std::max(r, std::max(g, b));
    float gap = v - std::min(r, std::min(g, b));
    
    if (gap > 0.0f) {
        if (r == v) {
            h = (g - b) / gap;
            if (h < 0.0f) h += 6.0f;
        } else if (g == v) {
            h = 2.0f + (b - r) / gap;
        } else {
            h = 4.0f + (r - g) / gap;
        }
        s = gap / v;
    } else {
        h = 0.0f;
        s = 0.0f;
    }
}

static inline void hsv_to_rgb(float h, float s, float v, float& r, float& g, float& b) {
    if (s > 0.0f) {
        h = fmodf(h, 6.0f);
        if (h < 0.0f) h += 6.0f;
        
        int i = (int)h;
        float f = h - (float)i;
        float p = v * (1.0f - s);
        float q = v * (1.0f - s * f);
        float t = v * (1.0f - s * (1.0f - f));
        
        switch (i) {
            case 0: r = v; g = t; b = p; break;
            case 1: r = q; g = v; b = p; break;
            case 2: r = p; g = v; b = t; break;
            case 3: r = p; g = q; b = v; break;
            case 4: r = t; g = p; b = v; break;
            case 5: r = v; g = p; b = q; break;
            case 6: r = v; g = t; b = p; break;  // Edge case
        }
    } else {
        r = v;
        g = v;
        b = v;
    }
}

//=============================================================================
// HueSatMap - 3D LUT for HSV adjustments (from dng_hue_sat_map.cpp)
//=============================================================================

struct HSBModify {
    float hue_shift;   // Hue shift in degrees
    float sat_scale;   // Saturation scale factor
    float val_scale;   // Value scale factor
};

// Apply HueSatMap to a single pixel
// map_data: flattened 3D array of HSBModify [val][hue][sat]
// hue_divs, sat_divs, val_divs: table dimensions
static void apply_hue_sat_map(
    float& r, float& g, float& b,
    const HSBModify* map_data,
    uint32_t hue_divs, uint32_t sat_divs, uint32_t val_divs
) {
    // Convert to HSV
    float h, s, v;
    r = std::max(0.0f, r);
    g = std::max(0.0f, g);
    b = std::max(0.0f, b);
    rgb_to_hsv(r, g, b, h, s, v);
    
    // Scale factors for indexing
    float h_scale = (hue_divs < 2) ? 0.0f : (hue_divs * (1.0f / 6.0f));
    float s_scale = (float)((int32_t)sat_divs - 1);
    float v_scale = (float)((int32_t)val_divs - 1);
    
    int32_t max_hue_idx = (int32_t)hue_divs - 1;
    int32_t max_sat_idx = (int32_t)sat_divs - 2;
    int32_t max_val_idx = (int32_t)val_divs - 2;
    
    int32_t hue_step = sat_divs;
    int32_t val_step = hue_divs * hue_step;
    
    float hue_shift, sat_scale, val_scale;
    
    if (val_divs < 2) {
        // 2.5D table (most common)
        float h_scaled = h * h_scale;
        float s_scaled = s * s_scale;
        
        int32_t h_idx0 = (int32_t)h_scaled;
        int32_t s_idx0 = (int32_t)s_scaled;
        s_idx0 = std::min(s_idx0, max_sat_idx);
        
        int32_t h_idx1 = h_idx0 + 1;
        if (h_idx0 >= max_hue_idx) {
            h_idx0 = max_hue_idx;
            h_idx1 = 0;
        }
        
        float h_fract1 = h_scaled - (float)h_idx0;
        float s_fract1 = s_scaled - (float)s_idx0;
        float h_fract0 = 1.0f - h_fract1;
        float s_fract0 = 1.0f - s_fract1;
        
        const HSBModify* e00 = map_data + h_idx0 * hue_step + s_idx0;
        const HSBModify* e01 = map_data + h_idx1 * hue_step + s_idx0;
        
        float hs0 = h_fract0 * e00->hue_shift + h_fract1 * e01->hue_shift;
        float ss0 = h_fract0 * e00->sat_scale + h_fract1 * e01->sat_scale;
        float vs0 = h_fract0 * e00->val_scale + h_fract1 * e01->val_scale;
        
        e00++; e01++;
        float hs1 = h_fract0 * e00->hue_shift + h_fract1 * e01->hue_shift;
        float ss1 = h_fract0 * e00->sat_scale + h_fract1 * e01->sat_scale;
        float vs1 = h_fract0 * e00->val_scale + h_fract1 * e01->val_scale;
        
        hue_shift = s_fract0 * hs0 + s_fract1 * hs1;
        sat_scale = s_fract0 * ss0 + s_fract1 * ss1;
        val_scale = s_fract0 * vs0 + s_fract1 * vs1;
    } else {
        // Full 3D table - trilinear interpolation
        float h_scaled = h * h_scale;
        float s_scaled = s * s_scale;
        float v_scaled = v * v_scale;
        
        int32_t h_idx0 = (int32_t)h_scaled;
        int32_t s_idx0 = (int32_t)s_scaled;
        int32_t v_idx0 = (int32_t)v_scaled;
        
        s_idx0 = std::min(s_idx0, max_sat_idx);
        v_idx0 = std::min(v_idx0, max_val_idx);
        
        int32_t h_idx1 = h_idx0 + 1;
        if (h_idx0 >= max_hue_idx) {
            h_idx0 = max_hue_idx;
            h_idx1 = 0;
        }
        
        float h_fract1 = h_scaled - (float)h_idx0;
        float s_fract1 = s_scaled - (float)s_idx0;
        float v_fract1 = v_scaled - (float)v_idx0;
        float h_fract0 = 1.0f - h_fract1;
        float s_fract0 = 1.0f - s_fract1;
        float v_fract0 = 1.0f - v_fract1;
        
        const HSBModify* e00 = map_data + v_idx0 * val_step + h_idx0 * hue_step + s_idx0;
        const HSBModify* e01 = map_data + v_idx0 * val_step + h_idx1 * hue_step + s_idx0;
        const HSBModify* e10 = e00 + val_step;
        const HSBModify* e11 = e01 + val_step;
        
        float hs0 = v_fract0 * (h_fract0 * e00->hue_shift + h_fract1 * e01->hue_shift) +
                    v_fract1 * (h_fract0 * e10->hue_shift + h_fract1 * e11->hue_shift);
        float ss0 = v_fract0 * (h_fract0 * e00->sat_scale + h_fract1 * e01->sat_scale) +
                    v_fract1 * (h_fract0 * e10->sat_scale + h_fract1 * e11->sat_scale);
        float vs0 = v_fract0 * (h_fract0 * e00->val_scale + h_fract1 * e01->val_scale) +
                    v_fract1 * (h_fract0 * e10->val_scale + h_fract1 * e11->val_scale);
        
        e00++; e01++; e10++; e11++;
        float hs1 = v_fract0 * (h_fract0 * e00->hue_shift + h_fract1 * e01->hue_shift) +
                    v_fract1 * (h_fract0 * e10->hue_shift + h_fract1 * e11->hue_shift);
        float ss1 = v_fract0 * (h_fract0 * e00->sat_scale + h_fract1 * e01->sat_scale) +
                    v_fract1 * (h_fract0 * e10->sat_scale + h_fract1 * e11->sat_scale);
        float vs1 = v_fract0 * (h_fract0 * e00->val_scale + h_fract1 * e01->val_scale) +
                    v_fract1 * (h_fract0 * e10->val_scale + h_fract1 * e11->val_scale);
        
        hue_shift = s_fract0 * hs0 + s_fract1 * hs1;
        sat_scale = s_fract0 * ss0 + s_fract1 * ss1;
        val_scale = s_fract0 * vs0 + s_fract1 * vs1;
    }
    
    // Apply adjustments
    hue_shift *= (6.0f / 360.0f);  // Convert degrees to H range
    h += hue_shift;
    s = std::min(s * sat_scale, 1.0f);
    v = std::max(0.0f, std::min(v * val_scale, 1.0f));
    
    // Convert back to RGB
    hsv_to_rgb(h, s, v, r, g, b);
}

//=============================================================================
// Dual-Illuminant Matrix Interpolation (from dng_color_spec.cpp)
// Interpolates ColorMatrix1/2, ForwardMatrix1/2, etc. by color temperature
//=============================================================================

// Calculate interpolation weight based on inverse temperature
// temp1, temp2: calibration illuminant temperatures (Kelvin)
// target_temp: white point temperature to interpolate for
// Returns weight for matrix1 (0.0 = use matrix2, 1.0 = use matrix1)
static double calculate_illuminant_weight(double temp1, double temp2, double target_temp) {
    if (target_temp <= temp1) return 1.0;
    if (target_temp >= temp2) return 0.0;
    
    // Interpolate in 1/T space (mired-like)
    double inv_t = 1.0 / target_temp;
    double inv_t1 = 1.0 / temp1;
    double inv_t2 = 1.0 / temp2;
    
    return (inv_t - inv_t2) / (inv_t1 - inv_t2);
}

// Interpolate two 3x3 matrices
static void interpolate_matrix_3x3(
    const double* m1, const double* m2, double weight, double* result
) {
    double w1 = weight;
    double w2 = 1.0 - weight;
    for (int i = 0; i < 9; i++) {
        result[i] = w1 * m1[i] + w2 * m2[i];
    }
}

// Multiply two 3x3 matrices: result = a * b
static void multiply_matrix_3x3(const double* a, const double* b, double* result) {
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            result[i * 3 + j] = 0.0;
            for (int k = 0; k < 3; k++) {
                result[i * 3 + j] += a[i * 3 + k] * b[k * 3 + j];
            }
        }
    }
}

// Invert a 3x3 matrix
static bool invert_matrix_3x3(const double* m, double* result) {
    double det = m[0] * (m[4] * m[8] - m[5] * m[7])
               - m[1] * (m[3] * m[8] - m[5] * m[6])
               + m[2] * (m[3] * m[7] - m[4] * m[6]);
    
    if (fabs(det) < 1e-10) return false;
    
    double inv_det = 1.0 / det;
    result[0] = (m[4] * m[8] - m[5] * m[7]) * inv_det;
    result[1] = (m[2] * m[7] - m[1] * m[8]) * inv_det;
    result[2] = (m[1] * m[5] - m[2] * m[4]) * inv_det;
    result[3] = (m[5] * m[6] - m[3] * m[8]) * inv_det;
    result[4] = (m[0] * m[8] - m[2] * m[6]) * inv_det;
    result[5] = (m[2] * m[3] - m[0] * m[5]) * inv_det;
    result[6] = (m[3] * m[7] - m[4] * m[6]) * inv_det;
    result[7] = (m[1] * m[6] - m[0] * m[7]) * inv_det;
    result[8] = (m[0] * m[4] - m[1] * m[3]) * inv_det;
    return true;
}

//=============================================================================
// Bradford Chromatic Adaptation (from dng_color_spec.cpp)
// Used for white point adaptation
//=============================================================================

static const double kBradfordMatrix[9] = {
     0.8951,  0.2664, -0.1614,
    -0.7502,  1.7135,  0.0367,
     0.0389, -0.0685,  1.0296
};

static const double kBradfordMatrixInv[9] = {
     0.9869929, -0.1470543,  0.1599627,
     0.4323053,  0.5183603,  0.0492912,
    -0.0085287,  0.0400428,  0.9684867
};

// Convert xy chromaticity to XYZ (Y=1)
static void xy_to_XYZ(double x, double y, double* XYZ) {
    XYZ[0] = x / y;
    XYZ[1] = 1.0;
    XYZ[2] = (1.0 - x - y) / y;
}

// Compute Bradford chromatic adaptation matrix from white1 to white2
static void compute_bradford_adaptation(
    double src_x, double src_y, double dst_x, double dst_y, double* result
) {
    double src_XYZ[3], dst_XYZ[3];
    xy_to_XYZ(src_x, src_y, src_XYZ);
    xy_to_XYZ(dst_x, dst_y, dst_XYZ);
    
    // Transform to cone response domain
    double src_cone[3], dst_cone[3];
    for (int i = 0; i < 3; i++) {
        src_cone[i] = kBradfordMatrix[i*3+0] * src_XYZ[0] +
                      kBradfordMatrix[i*3+1] * src_XYZ[1] +
                      kBradfordMatrix[i*3+2] * src_XYZ[2];
        dst_cone[i] = kBradfordMatrix[i*3+0] * dst_XYZ[0] +
                      kBradfordMatrix[i*3+1] * dst_XYZ[1] +
                      kBradfordMatrix[i*3+2] * dst_XYZ[2];
    }
    
    // Diagonal scaling matrix
    double scale[9] = {0};
    scale[0] = (src_cone[0] > 0) ? dst_cone[0] / src_cone[0] : 1.0;
    scale[4] = (src_cone[1] > 0) ? dst_cone[1] / src_cone[1] : 1.0;
    scale[8] = (src_cone[2] > 0) ? dst_cone[2] / src_cone[2] : 1.0;
    
    // Clamp scaling
    scale[0] = std::max(0.1, std::min(10.0, scale[0]));
    scale[4] = std::max(0.1, std::min(10.0, scale[4]));
    scale[8] = std::max(0.1, std::min(10.0, scale[8]));
    
    // Result = BradfordInv * Scale * Bradford
    double temp[9];
    multiply_matrix_3x3(scale, kBradfordMatrix, temp);
    multiply_matrix_3x3(kBradfordMatrixInv, temp, result);
}

//=============================================================================
// 1D Tone Curve Interpolation
//=============================================================================

// Interpolate a custom tone curve (for ProfileToneCurve)
static float interpolate_tone_curve(float x, const float* curve, int curve_size) {
    if (curve_size < 2) return x;
    
    x = std::max(0.0f, std::min(1.0f, x));
    float idx = x * (float)(curve_size - 1);
    int i = (int)idx;
    if (i >= curve_size - 1) i = curve_size - 2;
    
    float fract = idx - (float)i;
    return curve[i] * (1.0f - fract) + curve[i + 1] * fract;
}

//=============================================================================
// Stage 1: Pre-Demosaic Operations (on RAW CFA data)
//=============================================================================

// Apply linearization table to RAW data
// Converts sensor ADC values to linear light values
// table: LUT mapping input [0, max_val] to output
// max_val: maximum input value (e.g., 16383 for 14-bit)
static void apply_linearization_table(
    float* data, npy_intp count,
    const float* table, int table_size, float max_val
) {
    float scale = (float)(table_size - 1) / max_val;
    for (npy_intp i = 0; i < count; i++) {
        float val = data[i];
        float idx = val * scale;
        int i0 = (int)idx;
        if (i0 < 0) i0 = 0;
        if (i0 >= table_size - 1) i0 = table_size - 2;
        float fract = idx - (float)i0;
        data[i] = table[i0] * (1.0f - fract) + table[i0 + 1] * fract;
    }
}

// Normalize RAW data using black and white levels per DNG spec Chapter 5.
// Implements: linear = (raw - BlackLevel[r%rR][c%rC][s] - DeltaH[c] - DeltaV[r]) / (WhiteLevel[s] - BlackLevel[...])
//
// Parameters:
//   data: raw pixel data, shape (height, width, samples_per_pixel) or (height, width) if samples=1
//   black_level: 3D array [repeat_rows][repeat_cols][samples_per_pixel] stored row-major
//   black_repeat_rows, black_repeat_cols: dimensions of the repeating black level pattern
//   black_delta_h: per-column delta array, length=width (or NULL if not used)
//   black_delta_v: per-row delta array, length=height (or NULL if not used)
//   white_level: per-sample white level array, length=samples_per_pixel
//   samples_per_pixel: number of samples (1 for CFA, 3 for LinearRaw)
//
// SDK ref: dng_linearize_plane.cpp, dng_linearization_info
static void normalize_black_white(
    float* data, npy_intp height, npy_intp width, int samples_per_pixel,
    const float* black_level, int black_repeat_rows, int black_repeat_cols,
    const float* black_delta_h, npy_intp delta_h_count,
    const float* black_delta_v, npy_intp delta_v_count,
    const float* white_level, int white_count,
    const uint16_t* linearization_table = nullptr, int linearization_table_size = 0
) {
    for (npy_intp row = 0; row < height; row++) {
        // Get row delta (0 if not provided)
        float delta_v = 0.0f;
        if (black_delta_v != NULL && delta_v_count > 0) {
            delta_v = black_delta_v[row % delta_v_count];
        }
        
        // Black level row index in repeating pattern
        int black_row = (int)(row % black_repeat_rows);
        
        for (npy_intp col = 0; col < width; col++) {
            // Get column delta (0 if not provided)
            float delta_h = 0.0f;
            if (black_delta_h != NULL && delta_h_count > 0) {
                delta_h = black_delta_h[col % delta_h_count];
            }
            
            // Black level column index in repeating pattern
            int black_col = (int)(col % black_repeat_cols);
            
            for (int sample = 0; sample < samples_per_pixel; sample++) {
                npy_intp pixel_idx = (row * width + col) * samples_per_pixel + sample;
                
                // Apply LinearizationTable if present (before black/white normalization)
                // SDK ref: dng_linearize_plane::Process - LUT lookup on raw ADC values
                float pixel_val = data[pixel_idx];
                if (linearization_table != nullptr && linearization_table_size > 0) {
                    int lut_idx = (int)pixel_val;
                    // Clamp to table bounds (upper bound common when table < max pixel value)
                    if (lut_idx >= linearization_table_size) lut_idx = linearization_table_size - 1;
                    pixel_val = (float)linearization_table[lut_idx];
                }
                
                // BlackLevel index: [row][col][sample] in row-major order
                int black_idx = (black_row * black_repeat_cols + black_col) * samples_per_pixel + sample;
                float black = black_level[black_idx];
                
                // WhiteLevel per sample
                float white = white_level[sample % white_count];
                
                // Total black level including deltas
                float total_black = black + delta_h + delta_v;
                
                // Normalize to [0, 1]
                float range = white - total_black;
                if (range > 0.0f) {
                    data[pixel_idx] = (pixel_val - total_black) / range;
                } else {
                    data[pixel_idx] = 0.0f;
                }
                data[pixel_idx] = std::max(0.0f, std::min(1.0f, data[pixel_idx]));
            }
        }
    }
}

//=============================================================================
// Stage 2: Post-Demosaic Operations (on RGB data)
//=============================================================================

// SDK ref: dng_resample.h lines 192-194
const uint32_t kResampleSubsampleBits2D = 5;
const uint32_t kResampleSubsampleCount2D = 1 << kResampleSubsampleBits2D;  // 32

// SDK ref: dng_resample.cpp dng_resample_bicubic::Evaluate lines 33-48
static inline double dng_resample_bicubic_Evaluate(double x) {
    const double A = -0.75;
    x = std::abs(x);
    
    if (x >= 2.0)
        return 0.0;
    else if (x >= 1.0)
        return (((A * x - 5.0 * A) * x + 8.0 * A) * x - 4.0 * A);
    else
        return (((A + 2.0) * x - (A + 3.0)) * x * x + 1.0);
}

// SDK ref: dng_resample.cpp dng_resample_weights_2d::Initialize lines 308-441
// Precomputed 2D bicubic weights: 32x32 fractional positions, 4x4 weights each
// Total: 32 * 32 * 16 = 16384 weights
static float* g_bicubic_weights_2d = nullptr;
static const uint32_t kBicubicWidth = 4;  // 2 * radius where radius = 2
static const uint32_t kBicubicWidthSqr = 16;

static void init_bicubic_weights_2d() {
    if (g_bicubic_weights_2d != nullptr) return;
    
    g_bicubic_weights_2d = new float[kResampleSubsampleCount2D * kResampleSubsampleCount2D * kBicubicWidthSqr];
    
    // SDK ref: dng_resample.cpp lines 369-441
    for (uint32_t y = 0; y < kResampleSubsampleCount2D; y++) {
        double yFract = y * (1.0 / (double)kResampleSubsampleCount2D);
        
        for (uint32_t x = 0; x < kResampleSubsampleCount2D; x++) {
            double xFract = x * (1.0 / (double)kResampleSubsampleCount2D);
            
            float* w32 = g_bicubic_weights_2d + 
                         (y * kResampleSubsampleCount2D + x) * kBicubicWidthSqr;
            
            // SDK ref: lines 386-428
            double t32 = 0.0;
            uint32_t index = 0;
            
            for (uint32_t i = 0; i < kBicubicWidth; i++) {
                int32_t yInt = ((int32_t)i) - 2 + 1;  // fRadius = 2, so -1, 0, 1, 2
                double yPos = yInt - yFract;
                
                for (uint32_t j = 0; j < kBicubicWidth; j++) {
                    int32_t xInt = ((int32_t)j) - 2 + 1;  // -1, 0, 1, 2
                    double xPos = xInt - xFract;
                    
                    // SDK ref: lines 415-418 - Separable kernel
                    w32[index] = (float)(dng_resample_bicubic_Evaluate(xPos) *
                                         dng_resample_bicubic_Evaluate(yPos));
                    t32 += w32[index];
                    index++;
                }
            }
            
            // SDK ref: lines 430-438 - Normalize weights to sum to 1.0
            float s32 = (float)(1.0 / t32);
            for (uint32_t i = 0; i < kBicubicWidthSqr; i++) {
                w32[i] *= s32;
            }
        }
    }
}

// WarpRectilinear lens distortion correction with per-plane coefficients
// SDK ref: dng_lens_correction.cpp dng_filter_warp::GetSrcPixelPosition, EvaluateRatio
// Uses radial polynomial model: ratio = kr0 + kr2*r^2 + kr4*r^4 + kr6*r^6 (EVEN powers)
// Each color plane has its own coefficients for lateral CA correction
// center_x, center_y: optical center in normalized [0,1] coordinates
static void warp_rectilinear(
    const float* src, float* dst,
    npy_intp height, npy_intp width, int channels,
    const double* radial_params, int num_planes, int num_coeffs,  // [num_planes][num_coeffs]
    const double* tangential_params,  // [num_planes][2] or NULL
    double center_x, double center_y,
    bool use_bicubic = true  // SDK ref: dng_lens_correction.cpp line 1251 uses dng_resample_bicubic
) {
    // SDK ref: dng_lens_correction.cpp line 1253 - Initialize weights
    if (use_bicubic) {
        init_bicubic_weights_2d();
    }
    
    // SDK ref: dng_lens_correction.cpp lines 1200-1236
    double cx = center_x * width;
    double cy = center_y * height;
    
    // SDK: fNormRadius = MaxDistancePointToRect(squareCenter, squareBounds)
    double corner_dists[4] = {
        std::sqrt(cx*cx + cy*cy),
        std::sqrt((width-cx)*(width-cx) + cy*cy),
        std::sqrt(cx*cx + (height-cy)*(height-cy)),
        std::sqrt((width-cx)*(width-cx) + (height-cy)*(height-cy))
    };
    double fNormRadius = *std::max_element(corner_dists, corner_dists + 4);
    if (fNormRadius < 1.0) fNormRadius = 1.0;
    double fInvNormRadius = 1.0 / fNormRadius;
    
    for (npy_intp y = 0; y < height; y++) {
        for (npy_intp x = 0; x < width; x++) {
            // SDK ref: GetSrcPixelPosition lines 1601-1604
            double diff_h = (double)x - cx;
            double diff_v = (double)y - cy;
            double diffNorm_h = diff_h * fInvNormRadius;
            double diffNorm_v = diff_v * fInvNormRadius;
            
            // SDK: rr = Min(diffNormSqr.v + diffNormSqr.h, 1.0)
            double rr = diffNorm_h * diffNorm_h + diffNorm_v * diffNorm_v;
            if (rr > 1.0) rr = 1.0;
            
            npy_intp dst_idx = (y * width + x) * channels;
            
            // SDK applies different warp per color plane for lateral CA correction
            for (int c = 0; c < channels && c < num_planes; c++) {
                const double* plane_radial = radial_params + c * num_coeffs;
                const double* plane_tan = tangential_params ? tangential_params + c * 2 : NULL;
                
                // SDK ref: WarpRectilinear v1 stores coeffs for EVEN powers (r^0, r^2, r^4, r^6)
                // dng_lens_correction.cpp lines 1908-1911: fData[0], fData[2], fData[4], fData[6]
                // Since rr = r^2, polynomial is: kr0 + kr2*rr + kr4*rr^2 + kr6*rr^3
                double ratio = plane_radial[0];
                double r_pow = rr;
                for (int i = 1; i < num_coeffs && i < 4; i++) {
                    ratio += plane_radial[i] * r_pow;
                    r_pow *= rr;
                }
                
                // SDK ref: GetSrcPixelPosition lines 1625-1626
                double dSrc_h = diff_h * ratio;
                double dSrc_v = diff_v * ratio;
                
                // Apply tangential warp if provided
                if (plane_tan && (plane_tan[0] != 0.0 || plane_tan[1] != 0.0)) {
                    double kt0 = plane_tan[0];
                    double kt1 = plane_tan[1];
                    double tan_h = 2*kt0*diffNorm_h*diffNorm_v + kt1*(rr + 2*diffNorm_h*diffNorm_h);
                    double tan_v = 2*kt1*diffNorm_h*diffNorm_v + kt0*(rr + 2*diffNorm_v*diffNorm_v);
                    dSrc_h += fNormRadius * tan_h;
                    dSrc_v += fNormRadius * tan_v;
                }
                
                // SDK ref: line 1663
                double src_x = cx + dSrc_h;
                double src_y = cy + dSrc_v;
                
                // SDK ref: lines 1511-1514 - clamp to image bounds
                src_x = std::max(0.0, std::min(src_x, (double)(width - 1)));
                src_y = std::max(0.0, std::min(src_y, (double)(height - 1)));
                
                if (use_bicubic) {
                    // SDK ref: dng_lens_correction.cpp lines 1516-1577
                    // Decompose into integer and fractional parts
                    int32_t sInt_v = (int32_t)std::floor(src_y);
                    int32_t sInt_h = (int32_t)std::floor(src_x);
                    
                    // SDK ref: line 1521-1522 - fractional part scaled to subsample count
                    int32_t sFct_v = (int32_t)((src_y - (double)sInt_v) * kResampleSubsampleCount2D);
                    int32_t sFct_h = (int32_t)((src_x - (double)sInt_h) * kResampleSubsampleCount2D);
                    
                    // SDK ref: line 1526 - add resample offset (1 - fRadius = 1 - 2 = -1)
                    sInt_v += -1;
                    sInt_h += -1;
                    
                    // SDK ref: lines 1530-1552 - clip
                    int32_t hMin = 0;
                    int32_t hMax = (int32_t)(width - kBicubicWidth);
                    int32_t vMin = 0;
                    int32_t vMax = (int32_t)(height - kBicubicWidth);
                    
                    if (sInt_h < hMin) { sInt_h = hMin; sFct_h = 0; }
                    else if (sInt_h > hMax) { sInt_h = hMax; sFct_h = 0; }
                    if (sInt_v < vMin) { sInt_v = vMin; sFct_v = 0; }
                    else if (sInt_v > vMax) { sInt_v = vMax; sFct_v = 0; }
                    
                    // SDK ref: line 1556 - get precomputed weights
                    const float* w = g_bicubic_weights_2d + 
                                     (sFct_v * kResampleSubsampleCount2D + sFct_h) * kBicubicWidthSqr;
                    
                    // SDK ref: lines 1562-1577 - perform 2D resample
                    float total = 0.0f;
                    int32_t wIdx = 0;
                    for (int32_t i = 0; i < (int32_t)kBicubicWidth; i++) {
                        for (int32_t j = 0; j < (int32_t)kBicubicWidth; j++) {
                            int32_t sy = sInt_v + i;
                            int32_t sx = sInt_h + j;
                            total += w[wIdx] * src[(sy * width + sx) * channels + c];
                            wIdx++;
                        }
                    }
                    
                    // SDK ref: line 1581 - Pin_real32
                    dst[dst_idx + c] = std::max(0.0f, std::min(1.0f, total));
                } else {
                    // Bilinear interpolation (fallback)
                    int x0 = (int)std::floor(src_x);
                    int y0 = (int)std::floor(src_y);
                    int x1 = std::min(x0 + 1, (int)(width - 1));
                    int y1 = std::min(y0 + 1, (int)(height - 1));
                    double fx = src_x - x0;
                    double fy = src_y - y0;
                    
                    double v00 = src[(y0 * width + x0) * channels + c];
                    double v01 = src[(y0 * width + x1) * channels + c];
                    double v10 = src[(y1 * width + x0) * channels + c];
                    double v11 = src[(y1 * width + x1) * channels + c];
                    
                    dst[dst_idx + c] = (float)(
                        v00 * (1-fx) * (1-fy) +
                        v01 * fx * (1-fy) +
                        v10 * (1-fx) * fy +
                        v11 * fx * fy
                    );
                }
            }
        }
    }
}

// Radial vignette correction
// Applies gain = 1 + k0*r^2 + k1*r^4 + k2*r^6 + k3*r^8 + k4*r^10
static void fix_vignette_radial(
    float* data, npy_intp height, npy_intp width, int channels,
    const double* params, int num_params,
    double center_x, double center_y
) {
    double cx = center_x * width;
    double cy = center_y * height;
    double corner_dists[4] = {
        std::sqrt(cx*cx + cy*cy),
        std::sqrt((width-cx)*(width-cx) + cy*cy),
        std::sqrt(cx*cx + (height-cy)*(height-cy)),
        std::sqrt((width-cx)*(width-cx) + (height-cy)*(height-cy))
    };
    double max_dist = *std::max_element(corner_dists, corner_dists + 4);
    if (max_dist < 1.0) max_dist = 1.0;
    
    for (npy_intp y = 0; y < height; y++) {
        for (npy_intp x = 0; x < width; x++) {
            double dx = ((double)x - cx) / max_dist;
            double dy = ((double)y - cy) / max_dist;
            double r2 = dx*dx + dy*dy;
            
            // Evaluate polynomial: gain = 1 + k0*r^2 + k1*r^4 + ...
            double gain = 1.0;
            double r2p = r2;
            for (int i = 0; i < num_params && i < 5; i++) {
                gain += params[i] * r2p;
                r2p *= r2;
            }
            
            // Apply gain
            npy_intp idx = (y * width + x) * channels;
            for (int c = 0; c < channels; c++) {
                data[idx + c] *= (float)gain;
            }
        }
    }
}

// Apply gain map (flat-field correction) to CFA data
// gain_map: 2D or 4D array of per-pixel gains
// For Bayer CFA: can be (H/2, W/2, 4) for 2x2 pattern or (H, W) for single plane
static void apply_gain_map_cfa(
    float* data, npy_intp height, npy_intp width,
    const float* gain_map, npy_intp gain_h, npy_intp gain_w,
    int cfa_pattern_width, int cfa_pattern_height
) {
    // Scale gain map coordinates to data coordinates
    float scale_y = (float)gain_h / (float)height;
    float scale_x = (float)gain_w / (float)width;
    
    for (npy_intp y = 0; y < height; y++) {
        for (npy_intp x = 0; x < width; x++) {
            // Bilinear interpolation of gain map
            float gy = y * scale_y;
            float gx = x * scale_x;
            
            int gy0 = (int)gy;
            int gx0 = (int)gx;
            if (gy0 >= gain_h - 1) gy0 = gain_h - 2;
            if (gx0 >= gain_w - 1) gx0 = gain_w - 2;
            
            float fy = gy - gy0;
            float fx = gx - gx0;
            
            float g00 = gain_map[gy0 * gain_w + gx0];
            float g01 = gain_map[gy0 * gain_w + gx0 + 1];
            float g10 = gain_map[(gy0 + 1) * gain_w + gx0];
            float g11 = gain_map[(gy0 + 1) * gain_w + gx0 + 1];
            
            float gain = g00 * (1-fx) * (1-fy) + g01 * fx * (1-fy) +
                        g10 * (1-fx) * fy + g11 * fx * fy;
            
            data[y * width + x] *= gain;
        }
    }
}

//=============================================================================
// sRGB Gamma Encoding
//=============================================================================

static inline float srgb_gamma(float linear) {
    if (linear <= 0.0031308f) {
        return linear * 12.92f;
    } else {
        return 1.055f * powf(linear, 1.0f / 2.4f) - 0.055f;
    }
}

//=============================================================================
// Python Module Functions
//=============================================================================

static PyObject* dng_color_temp_to_xy(PyObject* self, PyObject* args) {
    double temperature, tint;
    
    if (!PyArg_ParseTuple(args, "dd", &temperature, &tint)) {
        return NULL;
    }
    
    if (temperature < 1667.0 || temperature > 25000.0) {
        PyErr_SetString(PyExc_ValueError, "Temperature must be between 1667 and 25000 Kelvin");
        return NULL;
    }
    
    double x, y;
    temp_tint_to_xy(temperature, tint, &x, &y);
    
    return Py_BuildValue("(dd)", x, y);
}

static PyObject* dng_color_xy_to_temp(PyObject* self, PyObject* args) {
    double x, y;
    
    if (!PyArg_ParseTuple(args, "dd", &x, &y)) {
        return NULL;
    }
    
    if (x <= 0.0 || x >= 1.0 || y <= 0.0 || y >= 1.0) {
        PyErr_SetString(PyExc_ValueError, "xy coordinates must be in range (0, 1)");
        return NULL;
    }
    
    double temperature, tint;
    xy_to_temp_tint(x, y, &temperature, &tint);
    
    return Py_BuildValue("(dd)", temperature, tint);
}

static PyObject* dng_color_get_acr3_curve(PyObject* self, PyObject* args) {
    int num_points = 256;
    
    if (!PyArg_ParseTuple(args, "|i", &num_points)) {
        return NULL;
    }
    
    if (num_points < 2 || num_points > 65536) {
        PyErr_SetString(PyExc_ValueError, "num_points must be between 2 and 65536");
        return NULL;
    }
    
    npy_intp dims[1] = {num_points};
    PyObject* result = PyArray_SimpleNew(1, dims, NPY_FLOAT32);
    if (!result) return NULL;
    
    float* data = (float*)PyArray_DATA((PyArrayObject*)result);
    
    for (int i = 0; i < num_points; i++) {
        float x = (float)i / (float)(num_points - 1);
        data[i] = evaluate_acr3_curve(x);
    }
    
    return result;
}

// Apply HueSatMap to RGB image
static PyObject* dng_color_apply_hue_sat_map(PyObject* self, PyObject* args) {
    PyArrayObject* rgb_array = NULL;
    PyArrayObject* map_array = NULL;
    int hue_divs, sat_divs, val_divs;
    
    if (!PyArg_ParseTuple(args, "O!O!iii",
            &PyArray_Type, &rgb_array,
            &PyArray_Type, &map_array,
            &hue_divs, &sat_divs, &val_divs)) {
        return NULL;
    }
    
    // Validate RGB array
    if (PyArray_NDIM(rgb_array) != 3 || PyArray_DIM(rgb_array, 2) != 3) {
        PyErr_SetString(PyExc_ValueError, "rgb must be shape (H, W, 3)");
        return NULL;
    }
    if (PyArray_TYPE(rgb_array) != NPY_FLOAT32) {
        PyErr_SetString(PyExc_TypeError, "rgb must be float32");
        return NULL;
    }
    
    // Validate map array - should be (val_divs, hue_divs, sat_divs, 3)
    // or flattened with 3 values per entry (hue_shift, sat_scale, val_scale)
    npy_intp expected_entries = (npy_intp)hue_divs * sat_divs * val_divs;
    npy_intp map_entries = PyArray_SIZE(map_array) / 3;
    if (map_entries != expected_entries) {
        PyErr_Format(PyExc_ValueError, 
            "Map size mismatch: expected %ld entries, got %ld",
            (long)expected_entries, (long)map_entries);
        return NULL;
    }
    
    npy_intp height = PyArray_DIM(rgb_array, 0);
    npy_intp width = PyArray_DIM(rgb_array, 1);
    
    // Ensure contiguous input
    PyArrayObject* rgb_cont = (PyArrayObject*)PyArray_ContiguousFromAny(
        (PyObject*)rgb_array, NPY_FLOAT32, 3, 3);
    if (!rgb_cont) return NULL;
    
    PyArrayObject* map_cont = (PyArrayObject*)PyArray_ContiguousFromAny(
        (PyObject*)map_array, NPY_FLOAT32, 1, 4);
    if (!map_cont) {
        Py_DECREF(rgb_cont);
        return NULL;
    }
    
    // Create output array
    npy_intp dims[3] = {height, width, 3};
    PyObject* result = PyArray_SimpleNew(3, dims, NPY_FLOAT32);
    if (!result) {
        Py_DECREF(rgb_cont);
        Py_DECREF(map_cont);
        return NULL;
    }
    
    float* src_data = (float*)PyArray_DATA(rgb_cont);
    float* dst_data = (float*)PyArray_DATA((PyArrayObject*)result);
    const HSBModify* map_data = (const HSBModify*)PyArray_DATA(map_cont);
    
    // Copy source to dest first
    memcpy(dst_data, src_data, height * width * 3 * sizeof(float));
    
    // Apply HueSatMap to each pixel
    npy_intp total_pixels = height * width;
    for (npy_intp p = 0; p < total_pixels; p++) {
        npy_intp idx = p * 3;
        apply_hue_sat_map(
            dst_data[idx + 0], dst_data[idx + 1], dst_data[idx + 2],
            map_data, hue_divs, sat_divs, val_divs
        );
    }
    
    Py_DECREF(rgb_cont);
    Py_DECREF(map_cont);
    return result;
}

// Interpolate dual-illuminant matrices by color temperature
static PyObject* dng_color_interpolate_matrices(PyObject* self, PyObject* args) {
    PyArrayObject* matrix1_array = NULL;
    PyArrayObject* matrix2_array = NULL;
    double temp1, temp2, target_temp;
    
    if (!PyArg_ParseTuple(args, "O!O!ddd",
            &PyArray_Type, &matrix1_array,
            &PyArray_Type, &matrix2_array,
            &temp1, &temp2, &target_temp)) {
        return NULL;
    }
    
    // Validate arrays are 3x3
    if (PyArray_SIZE(matrix1_array) != 9 || PyArray_SIZE(matrix2_array) != 9) {
        PyErr_SetString(PyExc_ValueError, "Matrices must be 3x3 (9 elements)");
        return NULL;
    }
    
    PyArrayObject* m1_cont = (PyArrayObject*)PyArray_ContiguousFromAny(
        (PyObject*)matrix1_array, NPY_FLOAT64, 1, 2);
    PyArrayObject* m2_cont = (PyArrayObject*)PyArray_ContiguousFromAny(
        (PyObject*)matrix2_array, NPY_FLOAT64, 1, 2);
    
    if (!m1_cont || !m2_cont) {
        Py_XDECREF(m1_cont);
        Py_XDECREF(m2_cont);
        return NULL;
    }
    
    double* m1 = (double*)PyArray_DATA(m1_cont);
    double* m2 = (double*)PyArray_DATA(m2_cont);
    
    // Ensure temp1 < temp2
    if (temp1 > temp2) {
        std::swap(temp1, temp2);
        std::swap(m1, m2);
    }
    
    double weight = calculate_illuminant_weight(temp1, temp2, target_temp);
    
    npy_intp dims[2] = {3, 3};
    PyObject* result = PyArray_SimpleNew(2, dims, NPY_FLOAT64);
    if (!result) {
        Py_DECREF(m1_cont);
        Py_DECREF(m2_cont);
        return NULL;
    }
    
    double* result_data = (double*)PyArray_DATA((PyArrayObject*)result);
    interpolate_matrix_3x3(m1, m2, weight, result_data);
    
    Py_DECREF(m1_cont);
    Py_DECREF(m2_cont);
    return result;
}

// Apply sRGB gamma encoding (linear to sRGB)
// sRGB spec: if x <= 0.0031308: 12.92 * x, else: 1.055 * x^(1/2.4) - 0.055
static PyObject* dng_color_srgb_gamma(PyObject* self, PyObject* args) {
    PyArrayObject* rgb_array = NULL;
    
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &rgb_array)) {
        return NULL;
    }
    
    if (PyArray_NDIM(rgb_array) != 3 || PyArray_DIM(rgb_array, 2) != 3) {
        PyErr_SetString(PyExc_ValueError, "rgb must be shape (H, W, 3)");
        return NULL;
    }
    if (PyArray_TYPE(rgb_array) != NPY_FLOAT32) {
        PyErr_SetString(PyExc_TypeError, "rgb must be float32");
        return NULL;
    }
    
    npy_intp height = PyArray_DIM(rgb_array, 0);
    npy_intp width = PyArray_DIM(rgb_array, 1);
    
    PyArrayObject* rgb_cont = (PyArrayObject*)PyArray_ContiguousFromAny(
        (PyObject*)rgb_array, NPY_FLOAT32, 3, 3);
    if (!rgb_cont) return NULL;
    
    npy_intp dims[3] = {height, width, 3};
    PyObject* result = PyArray_SimpleNew(3, dims, NPY_FLOAT32);
    if (!result) {
        Py_DECREF(rgb_cont);
        return NULL;
    }
    
    float* src_data = (float*)PyArray_DATA(rgb_cont);
    float* dst_data = (float*)PyArray_DATA((PyArrayObject*)result);
    
    npy_intp total = height * width * 3;
    const float threshold = 0.0031308f;
    const float inv_gamma = 1.0f / 2.4f;
    
    for (npy_intp i = 0; i < total; i++) {
        float x = src_data[i];
        float y;
        if (x <= threshold) {
            y = 12.92f * x;
        } else {
            y = 1.055f * std::pow(x, inv_gamma) - 0.055f;
        }
        dst_data[i] = std::max(0.0f, std::min(1.0f, y));
    }
    
    Py_DECREF(rgb_cont);
    return result;
}

// Apply 3x3 color matrix transform to RGB image with optional clipping
static PyObject* dng_color_matrix_transform(PyObject* self, PyObject* args, PyObject* kwargs) {
    PyArrayObject* rgb_array = NULL;
    PyArrayObject* matrix_array = NULL;
    int clip = 1;  // Default: clip to [0,1]
    
    static char* kwlist[] = {"rgb", "matrix", "clip", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!O!|p", kwlist,
            &PyArray_Type, &rgb_array,
            &PyArray_Type, &matrix_array,
            &clip)) {
        return NULL;
    }
    
    if (PyArray_NDIM(rgb_array) != 3 || PyArray_DIM(rgb_array, 2) != 3) {
        PyErr_SetString(PyExc_ValueError, "rgb must be shape (H, W, 3)");
        return NULL;
    }
    if (PyArray_TYPE(rgb_array) != NPY_FLOAT32) {
        PyErr_SetString(PyExc_TypeError, "rgb must be float32");
        return NULL;
    }
    if (PyArray_SIZE(matrix_array) != 9) {
        PyErr_SetString(PyExc_ValueError, "matrix must be 3x3 (9 elements)");
        return NULL;
    }
    
    npy_intp height = PyArray_DIM(rgb_array, 0);
    npy_intp width = PyArray_DIM(rgb_array, 1);
    
    PyArrayObject* rgb_cont = (PyArrayObject*)PyArray_ContiguousFromAny(
        (PyObject*)rgb_array, NPY_FLOAT32, 3, 3);
    PyArrayObject* mat_cont = (PyArrayObject*)PyArray_ContiguousFromAny(
        (PyObject*)matrix_array, NPY_FLOAT32, 1, 2);
    
    if (!rgb_cont || !mat_cont) {
        Py_XDECREF(rgb_cont);
        Py_XDECREF(mat_cont);
        return NULL;
    }
    
    npy_intp dims[3] = {height, width, 3};
    PyObject* result = PyArray_SimpleNew(3, dims, NPY_FLOAT32);
    if (!result) {
        Py_DECREF(rgb_cont);
        Py_DECREF(mat_cont);
        return NULL;
    }
    
    float* src_data = (float*)PyArray_DATA(rgb_cont);
    float* dst_data = (float*)PyArray_DATA((PyArrayObject*)result);
    const float* m = (const float*)PyArray_DATA(mat_cont);
    
    // Matrix is row-major: m[row*3 + col]
    float m00 = m[0], m01 = m[1], m02 = m[2];
    float m10 = m[3], m11 = m[4], m12 = m[5];
    float m20 = m[6], m21 = m[7], m22 = m[8];
    
    npy_intp total_pixels = height * width;
    
    if (clip) {
        for (npy_intp i = 0; i < total_pixels; i++) {
            float r = src_data[i * 3 + 0];
            float g = src_data[i * 3 + 1];
            float b = src_data[i * 3 + 2];
            
            dst_data[i * 3 + 0] = std::max(0.0f, std::min(1.0f, m00 * r + m01 * g + m02 * b));
            dst_data[i * 3 + 1] = std::max(0.0f, std::min(1.0f, m10 * r + m11 * g + m12 * b));
            dst_data[i * 3 + 2] = std::max(0.0f, std::min(1.0f, m20 * r + m21 * g + m22 * b));
        }
    } else {
        for (npy_intp i = 0; i < total_pixels; i++) {
            float r = src_data[i * 3 + 0];
            float g = src_data[i * 3 + 1];
            float b = src_data[i * 3 + 2];
            
            dst_data[i * 3 + 0] = m00 * r + m01 * g + m02 * b;
            dst_data[i * 3 + 1] = m10 * r + m11 * g + m12 * b;
            dst_data[i * 3 + 2] = m20 * r + m21 * g + m22 * b;
        }
    }
    
    Py_DECREF(rgb_cont);
    Py_DECREF(mat_cont);
    return result;
}

// ============================================================================
// Exposure Ramp (dng_function_exposure_ramp from dng_render.cpp lines 50-103)
// Direct port of SDK code
// ============================================================================

// SDK ref: dng_render.cpp dng_function_exposure_ramp::Evaluate() lines 81-103
static inline float exposure_ramp_evaluate(float x, float black, float slope, 
                                           float radius, float qScale, 
                                           bool supportOverrange) {
    // Region 1: x <= black - radius → 0
    if (x <= black - radius)
        return 0.0f;
    
    // Region 2: x >= black + radius → linear ramp
    if (x >= black + radius) {
        float y = (x - black) * slope;
        if (!supportOverrange)
            y = std::min(y, 1.0f);
        return y;
    }
    
    // Region 3: quadratic blend
    float y = x - (black - radius);
    return qScale * y * y;
}

// Apply exposure ramp to RGB image
// SDK ref: dng_render.cpp lines 50-103, 1907-1928
static PyObject* dng_color_apply_exposure_ramp(PyObject* self, PyObject* args) {
    PyArrayObject* rgb_array = NULL;
    double white, black, minBlack;
    int supportOverrange = 0;
    
    if (!PyArg_ParseTuple(args, "O!ddd|p",
            &PyArray_Type, &rgb_array,
            &white, &black, &minBlack,
            &supportOverrange)) {
        return NULL;
    }
    
    if (PyArray_NDIM(rgb_array) != 3 || PyArray_DIM(rgb_array, 2) != 3) {
        PyErr_SetString(PyExc_ValueError, "rgb must be shape (H, W, 3)");
        return NULL;
    }
    if (PyArray_TYPE(rgb_array) != NPY_FLOAT32) {
        PyErr_SetString(PyExc_TypeError, "rgb must be float32");
        return NULL;
    }
    
    npy_intp height = PyArray_DIM(rgb_array, 0);
    npy_intp width = PyArray_DIM(rgb_array, 1);
    
    PyArrayObject* rgb_cont = (PyArrayObject*)PyArray_ContiguousFromAny(
        (PyObject*)rgb_array, NPY_FLOAT32, 3, 3);
    if (!rgb_cont) return NULL;
    
    npy_intp dims[3] = {height, width, 3};
    PyObject* result = PyArray_SimpleNew(3, dims, NPY_FLOAT32);
    if (!result) {
        Py_DECREF(rgb_cont);
        return NULL;
    }
    
    const float* src_data = (const float*)PyArray_DATA(rgb_cont);
    float* dst_data = (float*)PyArray_DATA((PyArrayObject*)result);
    
    // SDK ref: dng_render.cpp lines 55-75 (constructor)
    float slope = 1.0f / (float)(white - black);
    
    // Compute radius for quadratic blend region
    const float kMaxCurveX = 0.5f;      // Fraction of minBlack
    const float kMaxCurveY = 1.0f / 16.0f;  // Fraction of white
    
    float radius = std::min(kMaxCurveX * (float)minBlack, kMaxCurveY / slope);
    
    float qScale = 0.0f;
    if (radius > 0.0f)
        qScale = slope / (4.0f * radius);
    
    // Process all pixels
    npy_intp total = height * width * 3;
    for (npy_intp i = 0; i < total; i++) {
        dst_data[i] = exposure_ramp_evaluate(src_data[i], (float)black, slope, 
                                              radius, qScale, supportOverrange != 0);
    }
    
    Py_DECREF(rgb_cont);
    return result;
}

// ============================================================================
// Exposure Tone (dng_function_exposure_tone from dng_render.cpp lines 107-157)
// Applied AFTER tone curve for negative exposure compensation
// Direct port of SDK code
// ============================================================================

// SDK ref: dng_render.cpp dng_function_exposure_tone::Evaluate() lines 141-157
static inline float exposure_tone_evaluate(float x, float slope, float a, float b, float c) {
    // For negative exposure: darkens the image
    // Region 1: x <= 0.25 → linear darkening
    if (x <= 0.25f)
        return x * slope;
    
    // Region 2: x > 0.25 → quadratic (maps 1.0 → 1.0)
    return (a * x + b) * x + c;
}

// Apply exposure tone function to RGB image
// SDK ref: dng_render.cpp lines 107-157, 1009-1012
static PyObject* dng_color_apply_exposure_tone(PyObject* self, PyObject* args) {
    PyArrayObject* rgb_array = NULL;
    double exposure;
    
    if (!PyArg_ParseTuple(args, "O!d",
            &PyArray_Type, &rgb_array,
            &exposure)) {
        return NULL;
    }
    
    // SDK ref: line 109 - only apply for negative exposure
    if (exposure >= 0.0) {
        // Return input unchanged (NOP)
        Py_INCREF(rgb_array);
        return (PyObject*)rgb_array;
    }
    
    if (PyArray_NDIM(rgb_array) != 3 || PyArray_DIM(rgb_array, 2) != 3) {
        PyErr_SetString(PyExc_ValueError, "rgb must be shape (H, W, 3)");
        return NULL;
    }
    if (PyArray_TYPE(rgb_array) != NPY_FLOAT32) {
        PyErr_SetString(PyExc_TypeError, "rgb must be float32");
        return NULL;
    }
    
    npy_intp height = PyArray_DIM(rgb_array, 0);
    npy_intp width = PyArray_DIM(rgb_array, 1);
    
    PyArrayObject* rgb_cont = (PyArrayObject*)PyArray_ContiguousFromAny(
        (PyObject*)rgb_array, NPY_FLOAT32, 3, 3);
    if (!rgb_cont) return NULL;
    
    npy_intp dims[3] = {height, width, 3};
    PyObject* result = PyArray_SimpleNew(3, dims, NPY_FLOAT32);
    if (!result) {
        Py_DECREF(rgb_cont);
        return NULL;
    }
    
    float* src_data = (float*)PyArray_DATA(rgb_cont);
    float* dst_data = (float*)PyArray_DATA((PyArrayObject*)result);
    
    // SDK ref: dng_render.cpp lines 122-133
    // fSlope = pow(2.0, exposure)
    float slope = (float)pow(2.0, exposure);
    
    // Quadratic parameters: maps 1.0 → 1.0 with matching slope at crossover
    float a = 16.0f / 9.0f * (1.0f - slope);
    float b = slope - 0.5f * a;
    float c = 1.0f - a - b;
    
    // Process all pixels
    npy_intp total = height * width * 3;
    for (npy_intp i = 0; i < total; i++) {
        dst_data[i] = exposure_tone_evaluate(src_data[i], slope, a, b, c);
    }
    
    Py_DECREF(rgb_cont);
    return result;
}

// Apply hue-preserving RGB tone curve (RefBaselineRGBTone from dng_reference.cpp)
// This preserves color relationships by interpolating the middle channel
static PyObject* dng_color_apply_rgb_tone(PyObject* self, PyObject* args) {
    PyArrayObject* rgb_array = NULL;
    PyArrayObject* curve_array = NULL;
    
    if (!PyArg_ParseTuple(args, "O!O!",
            &PyArray_Type, &rgb_array,
            &PyArray_Type, &curve_array)) {
        return NULL;
    }
    
    if (PyArray_NDIM(rgb_array) != 3 || PyArray_DIM(rgb_array, 2) != 3) {
        PyErr_SetString(PyExc_ValueError, "rgb must be shape (H, W, 3)");
        return NULL;
    }
    if (PyArray_TYPE(rgb_array) != NPY_FLOAT32) {
        PyErr_SetString(PyExc_TypeError, "rgb must be float32");
        return NULL;
    }
    
    npy_intp height = PyArray_DIM(rgb_array, 0);
    npy_intp width = PyArray_DIM(rgb_array, 1);
    int curve_size = (int)PyArray_SIZE(curve_array);
    
    if (curve_size < 2) {
        PyErr_SetString(PyExc_ValueError, "Tone curve must have at least 2 points");
        return NULL;
    }
    
    PyArrayObject* rgb_cont = (PyArrayObject*)PyArray_ContiguousFromAny(
        (PyObject*)rgb_array, NPY_FLOAT32, 3, 3);
    PyArrayObject* curve_cont = (PyArrayObject*)PyArray_ContiguousFromAny(
        (PyObject*)curve_array, NPY_FLOAT32, 1, 1);
    
    if (!rgb_cont || !curve_cont) {
        Py_XDECREF(rgb_cont);
        Py_XDECREF(curve_cont);
        return NULL;
    }
    
    npy_intp dims[3] = {height, width, 3};
    PyObject* result = PyArray_SimpleNew(3, dims, NPY_FLOAT32);
    if (!result) {
        Py_DECREF(rgb_cont);
        Py_DECREF(curve_cont);
        return NULL;
    }
    
    float* src_data = (float*)PyArray_DATA(rgb_cont);
    float* dst_data = (float*)PyArray_DATA((PyArrayObject*)result);
    const float* curve = (const float*)PyArray_DATA(curve_cont);
    
    npy_intp total_pixels = height * width;
    
    // Process each pixel with hue-preserving tone mapping
    // SDK ref: dng_reference.cpp RefBaselineRGBTone lines 1868-1990
    for (npy_intp i = 0; i < total_pixels; i++) {
        float r = std::max(0.0f, std::min(1.0f, src_data[i * 3 + 0]));
        float g = std::max(0.0f, std::min(1.0f, src_data[i * 3 + 1]));
        float b = std::max(0.0f, std::min(1.0f, src_data[i * 3 + 2]));
        
        float rr, gg, bb;
        
        // Macro to apply curve and preserve hue for sorted r >= g >= b case
        #define RGBTone(r_in, g_in, b_in, r_out, g_out, b_out) \
            { \
                r_out = interpolate_tone_curve(r_in, curve, curve_size); \
                b_out = interpolate_tone_curve(b_in, curve, curve_size); \
                float denom = r_in - b_in; \
                if (denom > 1e-10f) { \
                    g_out = b_out + ((r_out - b_out) * (g_in - b_in) / denom); \
                } else { \
                    g_out = b_out; \
                } \
            }
        
        if (r >= g) {
            if (g > b) {
                // Case 1: r >= g > b
                RGBTone(r, g, b, rr, gg, bb);
            } else if (b > r) {
                // Case 2: b > r >= g
                RGBTone(b, r, g, bb, rr, gg);
            } else if (b > g) {
                // Case 3: r >= b > g
                RGBTone(r, b, g, rr, bb, gg);
            } else {
                // Case 4: r >= g == b
                rr = interpolate_tone_curve(r, curve, curve_size);
                gg = interpolate_tone_curve(g, curve, curve_size);
                bb = gg;
            }
        } else {
            if (r >= b) {
                // Case 5: g > r >= b
                RGBTone(g, r, b, gg, rr, bb);
            } else if (b > g) {
                // Case 6: b > g > r
                RGBTone(b, g, r, bb, gg, rr);
            } else {
                // Case 7: g >= b > r
                RGBTone(g, b, r, gg, bb, rr);
            }
        }
        
        #undef RGBTone
        
        dst_data[i * 3 + 0] = rr;
        dst_data[i * 3 + 1] = gg;
        dst_data[i * 3 + 2] = bb;
    }
    
    Py_DECREF(rgb_cont);
    Py_DECREF(curve_cont);
    return result;
}

// Apply custom tone curve to image (simple per-channel, no hue preservation)
static PyObject* dng_color_apply_tone_curve(PyObject* self, PyObject* args) {
    PyArrayObject* rgb_array = NULL;
    PyArrayObject* curve_array = NULL;
    
    if (!PyArg_ParseTuple(args, "O!O!",
            &PyArray_Type, &rgb_array,
            &PyArray_Type, &curve_array)) {
        return NULL;
    }
    
    if (PyArray_NDIM(rgb_array) != 3 || PyArray_DIM(rgb_array, 2) != 3) {
        PyErr_SetString(PyExc_ValueError, "rgb must be shape (H, W, 3)");
        return NULL;
    }
    if (PyArray_TYPE(rgb_array) != NPY_FLOAT32) {
        PyErr_SetString(PyExc_TypeError, "rgb must be float32");
        return NULL;
    }
    
    npy_intp height = PyArray_DIM(rgb_array, 0);
    npy_intp width = PyArray_DIM(rgb_array, 1);
    int curve_size = (int)PyArray_SIZE(curve_array);
    
    if (curve_size < 2) {
        PyErr_SetString(PyExc_ValueError, "Tone curve must have at least 2 points");
        return NULL;
    }
    
    PyArrayObject* rgb_cont = (PyArrayObject*)PyArray_ContiguousFromAny(
        (PyObject*)rgb_array, NPY_FLOAT32, 3, 3);
    PyArrayObject* curve_cont = (PyArrayObject*)PyArray_ContiguousFromAny(
        (PyObject*)curve_array, NPY_FLOAT32, 1, 1);
    
    if (!rgb_cont || !curve_cont) {
        Py_XDECREF(rgb_cont);
        Py_XDECREF(curve_cont);
        return NULL;
    }
    
    npy_intp dims[3] = {height, width, 3};
    PyObject* result = PyArray_SimpleNew(3, dims, NPY_FLOAT32);
    if (!result) {
        Py_DECREF(rgb_cont);
        Py_DECREF(curve_cont);
        return NULL;
    }
    
    float* src_data = (float*)PyArray_DATA(rgb_cont);
    float* dst_data = (float*)PyArray_DATA((PyArrayObject*)result);
    const float* curve = (const float*)PyArray_DATA(curve_cont);
    
    npy_intp total = height * width * 3;
    for (npy_intp i = 0; i < total; i++) {
        dst_data[i] = interpolate_tone_curve(src_data[i], curve, curve_size);
    }
    
    Py_DECREF(rgb_cont);
    Py_DECREF(curve_cont);
    return result;
}

// Compute Bradford chromatic adaptation matrix
static PyObject* dng_color_bradford_adapt(PyObject* self, PyObject* args) {
    double src_x, src_y, dst_x, dst_y;
    
    if (!PyArg_ParseTuple(args, "dddd", &src_x, &src_y, &dst_x, &dst_y)) {
        return NULL;
    }
    
    npy_intp dims[2] = {3, 3};
    PyObject* result = PyArray_SimpleNew(2, dims, NPY_FLOAT64);
    if (!result) return NULL;
    
    double* result_data = (double*)PyArray_DATA((PyArrayObject*)result);
    compute_bradford_adaptation(src_x, src_y, dst_x, dst_y, result_data);
    
    return result;
}

// Apply linearization table to RAW CFA data
static PyObject* dng_color_linearize(PyObject* self, PyObject* args) {
    PyArrayObject* data_array = NULL;
    PyArrayObject* table_array = NULL;
    float max_val;
    
    if (!PyArg_ParseTuple(args, "O!O!f",
            &PyArray_Type, &data_array,
            &PyArray_Type, &table_array,
            &max_val)) {
        return NULL;
    }
    
    if (PyArray_TYPE(data_array) != NPY_FLOAT32) {
        PyErr_SetString(PyExc_TypeError, "data must be float32");
        return NULL;
    }
    
    PyArrayObject* data_cont = (PyArrayObject*)PyArray_ContiguousFromAny(
        (PyObject*)data_array, NPY_FLOAT32, 1, 3);
    PyArrayObject* table_cont = (PyArrayObject*)PyArray_ContiguousFromAny(
        (PyObject*)table_array, NPY_FLOAT32, 1, 1);
    
    if (!data_cont || !table_cont) {
        Py_XDECREF(data_cont);
        Py_XDECREF(table_cont);
        return NULL;
    }
    
    // Copy data for output
    PyObject* result = PyArray_NewCopy(data_cont, NPY_CORDER);
    if (!result) {
        Py_DECREF(data_cont);
        Py_DECREF(table_cont);
        return NULL;
    }
    
    float* result_data = (float*)PyArray_DATA((PyArrayObject*)result);
    const float* table = (const float*)PyArray_DATA(table_cont);
    npy_intp count = PyArray_SIZE(data_cont);
    int table_size = (int)PyArray_SIZE(table_cont);
    
    apply_linearization_table(result_data, count, table, table_size, max_val);
    
    Py_DECREF(data_cont);
    Py_DECREF(table_cont);
    return result;
}

// Normalize RAW data using black/white levels per DNG spec Chapter 5.
// SDK ref: dng_linearize_plane.cpp, dng_linearization_info
//
// Args:
//   data: RAW pixel data, float32, shape (H, W) or (H, W, samples_per_pixel)
//   black_level: BlackLevel pattern, float32, shape (repeat_rows, repeat_cols, samples_per_pixel)
//                or flattened 1D array in row-col-sample order
//   black_repeat_rows: number of rows in repeating pattern (from BlackLevelRepeatDim[0])
//   black_repeat_cols: number of cols in repeating pattern (from BlackLevelRepeatDim[1])
//   samples_per_pixel: 1 for CFA, 3 for LinearRaw
//   white_level: WhiteLevel per sample, float32, shape (samples_per_pixel,)
//   black_delta_h: optional per-column delta, float32, shape (width,) or None
//   black_delta_v: optional per-row delta, float32, shape (height,) or None
static PyObject* dng_color_normalize_raw(PyObject* self, PyObject* args, PyObject* kwargs) {
    PyArrayObject* data_array = NULL;
    PyArrayObject* black_array = NULL;
    int black_repeat_rows = 1;
    int black_repeat_cols = 1;
    int samples_per_pixel = 1;
    PyArrayObject* white_array = NULL;
    PyObject* delta_h_obj = Py_None;
    PyObject* delta_v_obj = Py_None;
    PyObject* linearization_table_obj = Py_None;
    
    static const char* kwlist[] = {
        "data", "black_level", "black_repeat_rows", "black_repeat_cols",
        "samples_per_pixel", "white_level", "black_delta_h", "black_delta_v",
        "linearization_table", NULL
    };
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!O!iiiO!|OOO",
            const_cast<char**>(kwlist),
            &PyArray_Type, &data_array,
            &PyArray_Type, &black_array,
            &black_repeat_rows,
            &black_repeat_cols,
            &samples_per_pixel,
            &PyArray_Type, &white_array,
            &delta_h_obj,
            &delta_v_obj,
            &linearization_table_obj)) {
        return NULL;
    }
    
    // Validate data array
    if (PyArray_TYPE(data_array) != NPY_FLOAT32) {
        PyErr_SetString(PyExc_TypeError, "data must be float32");
        return NULL;
    }
    
    int ndim = PyArray_NDIM(data_array);
    if (ndim < 2 || ndim > 3) {
        PyErr_SetString(PyExc_ValueError, "data must be 2D (H,W) or 3D (H,W,C)");
        return NULL;
    }
    
    npy_intp height = PyArray_DIM(data_array, 0);
    npy_intp width = PyArray_DIM(data_array, 1);
    int data_samples = (ndim == 3) ? (int)PyArray_DIM(data_array, 2) : 1;
    
    if (data_samples != samples_per_pixel) {
        PyErr_SetString(PyExc_ValueError, "data channels must match samples_per_pixel");
        return NULL;
    }
    
    // Make contiguous copies
    PyArrayObject* data_cont = (PyArrayObject*)PyArray_ContiguousFromAny(
        (PyObject*)data_array, NPY_FLOAT32, 2, 3);
    PyArrayObject* black_cont = (PyArrayObject*)PyArray_ContiguousFromAny(
        (PyObject*)black_array, NPY_FLOAT32, 1, 3);
    PyArrayObject* white_cont = (PyArrayObject*)PyArray_ContiguousFromAny(
        (PyObject*)white_array, NPY_FLOAT32, 1, 1);
    
    if (!data_cont || !black_cont || !white_cont) {
        Py_XDECREF(data_cont);
        Py_XDECREF(black_cont);
        Py_XDECREF(white_cont);
        return NULL;
    }
    
    // Validate black_level size matches repeat pattern
    npy_intp expected_black_size = black_repeat_rows * black_repeat_cols * samples_per_pixel;
    if (PyArray_SIZE(black_cont) != expected_black_size) {
        PyErr_Format(PyExc_ValueError, 
            "black_level size (%zd) must equal repeat_rows * repeat_cols * samples_per_pixel (%zd)",
            (Py_ssize_t)PyArray_SIZE(black_cont), (Py_ssize_t)expected_black_size);
        Py_DECREF(data_cont);
        Py_DECREF(black_cont);
        Py_DECREF(white_cont);
        return NULL;
    }
    
    // Handle optional delta arrays
    PyArrayObject* delta_h_cont = NULL;
    PyArrayObject* delta_v_cont = NULL;
    npy_intp delta_h_count = 0;
    npy_intp delta_v_count = 0;
    
    if (delta_h_obj != Py_None && delta_h_obj != NULL) {
        delta_h_cont = (PyArrayObject*)PyArray_ContiguousFromAny(
            delta_h_obj, NPY_FLOAT32, 1, 1);
        if (!delta_h_cont) {
            Py_DECREF(data_cont);
            Py_DECREF(black_cont);
            Py_DECREF(white_cont);
            return NULL;
        }
        delta_h_count = PyArray_SIZE(delta_h_cont);
    }
    
    if (delta_v_obj != Py_None && delta_v_obj != NULL) {
        delta_v_cont = (PyArrayObject*)PyArray_ContiguousFromAny(
            delta_v_obj, NPY_FLOAT32, 1, 1);
        if (!delta_v_cont) {
            Py_DECREF(data_cont);
            Py_DECREF(black_cont);
            Py_DECREF(white_cont);
            Py_XDECREF(delta_h_cont);
            return NULL;
        }
        delta_v_count = PyArray_SIZE(delta_v_cont);
    }
    
    // Handle optional linearization table (uint16 LUT)
    PyArrayObject* lin_table_cont = NULL;
    int lin_table_size = 0;
    if (linearization_table_obj != Py_None && linearization_table_obj != NULL) {
        lin_table_cont = (PyArrayObject*)PyArray_ContiguousFromAny(
            linearization_table_obj, NPY_UINT16, 1, 1);
        if (!lin_table_cont) {
            Py_DECREF(data_cont);
            Py_DECREF(black_cont);
            Py_DECREF(white_cont);
            Py_XDECREF(delta_h_cont);
            Py_XDECREF(delta_v_cont);
            return NULL;
        }
        lin_table_size = (int)PyArray_SIZE(lin_table_cont);
    }
    
    // Copy data for output
    PyObject* result = PyArray_NewCopy(data_cont, NPY_CORDER);
    if (!result) {
        Py_DECREF(data_cont);
        Py_DECREF(black_cont);
        Py_DECREF(white_cont);
        Py_XDECREF(delta_h_cont);
        Py_XDECREF(delta_v_cont);
        Py_XDECREF(lin_table_cont);
        return NULL;
    }
    
    float* result_data = (float*)PyArray_DATA((PyArrayObject*)result);
    const float* black = (const float*)PyArray_DATA(black_cont);
    const float* white = (const float*)PyArray_DATA(white_cont);
    const float* delta_h = delta_h_cont ? (const float*)PyArray_DATA(delta_h_cont) : NULL;
    const float* delta_v = delta_v_cont ? (const float*)PyArray_DATA(delta_v_cont) : NULL;
    int white_count = (int)PyArray_SIZE(white_cont);
    
    const uint16_t* lin_table = lin_table_cont ? (const uint16_t*)PyArray_DATA(lin_table_cont) : nullptr;
    
    normalize_black_white(result_data, height, width, samples_per_pixel,
                         black, black_repeat_rows, black_repeat_cols,
                         delta_h, delta_h_count,
                         delta_v, delta_v_count,
                         white, white_count,
                         lin_table, lin_table_size);
    
    Py_DECREF(data_cont);
    Py_DECREF(black_cont);
    Py_DECREF(white_cont);
    Py_XDECREF(delta_h_cont);
    Py_XDECREF(delta_v_cont);
    Py_XDECREF(lin_table_cont);
    return result;
}

// Apply gain map (flat-field correction) to RAW CFA
static PyObject* dng_color_apply_gain_map(PyObject* self, PyObject* args) {
    PyArrayObject* data_array = NULL;
    PyArrayObject* gain_array = NULL;
    
    if (!PyArg_ParseTuple(args, "O!O!",
            &PyArray_Type, &data_array,
            &PyArray_Type, &gain_array)) {
        return NULL;
    }
    
    if (PyArray_TYPE(data_array) != NPY_FLOAT32) {
        PyErr_SetString(PyExc_TypeError, "data must be float32");
        return NULL;
    }
    if (PyArray_NDIM(data_array) != 2) {
        PyErr_SetString(PyExc_ValueError, "data must be 2D (H,W) CFA");
        return NULL;
    }
    if (PyArray_NDIM(gain_array) != 2) {
        PyErr_SetString(PyExc_ValueError, "gain_map must be 2D");
        return NULL;
    }
    
    npy_intp height = PyArray_DIM(data_array, 0);
    npy_intp width = PyArray_DIM(data_array, 1);
    npy_intp gain_h = PyArray_DIM(gain_array, 0);
    npy_intp gain_w = PyArray_DIM(gain_array, 1);
    
    PyArrayObject* data_cont = (PyArrayObject*)PyArray_ContiguousFromAny(
        (PyObject*)data_array, NPY_FLOAT32, 2, 2);
    PyArrayObject* gain_cont = (PyArrayObject*)PyArray_ContiguousFromAny(
        (PyObject*)gain_array, NPY_FLOAT32, 2, 2);
    
    if (!data_cont || !gain_cont) {
        Py_XDECREF(data_cont);
        Py_XDECREF(gain_cont);
        return NULL;
    }
    
    // Copy data for output
    PyObject* result = PyArray_NewCopy(data_cont, NPY_CORDER);
    if (!result) {
        Py_DECREF(data_cont);
        Py_DECREF(gain_cont);
        return NULL;
    }
    
    float* result_data = (float*)PyArray_DATA((PyArrayObject*)result);
    const float* gain = (const float*)PyArray_DATA(gain_cont);
    
    apply_gain_map_cfa(result_data, height, width, gain, gain_h, gain_w, 2, 2);
    
    Py_DECREF(data_cont);
    Py_DECREF(gain_cont);
    return result;
}

// ============================================================================
// ProfileGainTableMap application
// Direct port from DNG SDK dng_reference.cpp RefBaselineProfileGainTableMap()
// ============================================================================

static inline float Lerp_real32(float a, float b, float t) {
    return a + (b - a) * t;
}

static inline float Pin_real32(float lo, float x, float hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

static inline float Min_real32(float a, float b) {
    return a < b ? a : b;
}

static inline float Max_real32(float a, float b) {
    return a > b ? a : b;
}

static inline int Min_int32(int a, int b) {
    return a < b ? a : b;
}

// Direct port of RefBaselineProfileGainTableMap from dng_reference.cpp lines 3260-3460
static void RefBaselineProfileGainTableMap(
    const float* rSrcPtr,
    const float* gSrcPtr,
    const float* bSrcPtr,
    float* rDstPtr,
    float* gDstPtr,
    float* bDstPtr,
    const int cols,
    const int top,
    const int left,
    const int imageAreaL,
    const int imageAreaT,
    const int imageAreaW,
    const int imageAreaH,
    const float exposureWeightGain,
    // Gain table map parameters
    const int points_v,
    const int points_h,
    const float mapSpacingV,
    const float mapSpacingH,
    const float mapOriginV,
    const float mapOriginH,
    const int numTablePoints,
    const float* mapInputWeights,
    const float gamma,
    const float* gains,  // gains[row][col][tablePoint] flattened
    const bool supportOverrange
) {
    const float miw0 = mapInputWeights[0];
    const float miw1 = mapInputWeights[1];
    const float miw2 = mapInputWeights[2];
    const float miw3 = mapInputWeights[3];
    const float miw4 = mapInputWeights[4];
    
    const float mapOriginH32 = mapOriginH;
    const float mapOriginV32 = mapOriginV;
    const float mapSpacingH32 = mapSpacingH;
    const float mapSpacingV32 = mapSpacingV;
    
    const float xLimitLo = 0.0f;
    const float yLimitLo = 0.0f;
    const float xLimitHi = (float)(points_h - 1);
    const float yLimitHi = (float)(points_v - 1);
    
    const int xPixelLimit = points_h - 1;
    const int yPixelLimit = points_v - 1;
    
    const int tableSize = numTablePoints;
    const int tableLimit = tableSize - 1;
    
    // For gain table indexing: gains[row * rowStep + col * colStep + tableIdx]
    const int colStep = numTablePoints;
    const int rowStep = points_h * numTablePoints;
    
    // Initialize sample position. Note the half-pixel offset.
    float y = (float)top + 0.5f;
    float x = (float)left + 0.5f;
    
    // Process each pixel in this row.
    for (int col = 0; col < cols; col++) {
        
        // Transform to image-relative coordinates.
        float u_image = (x - (float)imageAreaL) / (float)imageAreaW;
        float v_image = (y - (float)imageAreaT) / (float)imageAreaH;
        
        // Transform to map-relative coordinates.
        float x_map = (u_image - mapOriginH32) / mapSpacingH32;
        float y_map = (v_image - mapOriginV32) / mapSpacingV32;
        
        // Clamp to valid sample positions.
        x_map = Pin_real32(xLimitLo, x_map, xLimitHi);
        y_map = Pin_real32(yLimitLo, y_map, yLimitHi);
        
        // Compute integer 2D indices.
        int x0 = (int)x_map;
        int x1 = Min_int32(x0 + 1, xPixelLimit);
        
        int y0 = (int)y_map;
        int y1 = Min_int32(y0 + 1, yPixelLimit);
        
        // Compute fractional weights.
        float xf = x_map - (float)x0;
        float yf = y_map - (float)y0;
        
        // Read linear RGB values in RIMM space.
        float r = rSrcPtr[col];
        float g = gSrcPtr[col];
        float b = bSrcPtr[col];
        
        // Apply MapInputWeights (5-element dot product).
        float minValue = Min_real32(r, Min_real32(g, b));
        float maxValue = Max_real32(r, Max_real32(g, b));
        
        float weight = ((miw0 * r) +
                       (miw1 * g) +
                       (miw2 * b) +
                       (miw3 * minValue) +
                       (miw4 * maxValue));
        
        // Scale weight by baseline exposure.
        weight = weight * exposureWeightGain;
        
        // Clamp weight to [0,1].
        weight = Pin_real32(0.0f, weight, 1.0f);
        
        // Apply gamma parameter.
        if (gamma != 1.0f)
            weight = powf(weight, gamma);
        
        // Scale weight by table size and compute table indices.
        float weightScaled = weight * (float)tableSize;
        
        int w0 = Min_int32((int)weightScaled, tableLimit);
        int w1 = Min_int32(w0 + 1, tableLimit);
        
        float wf = weightScaled - (float)w0;
        
        // Look up 8 gains.
        float gain000 = gains[y0 * rowStep + x0 * colStep + w0];
        float gain001 = gains[y0 * rowStep + x0 * colStep + w1];
        float gain010 = gains[y0 * rowStep + x1 * colStep + w0];
        float gain011 = gains[y0 * rowStep + x1 * colStep + w1];
        float gain100 = gains[y1 * rowStep + x0 * colStep + w0];
        float gain101 = gains[y1 * rowStep + x0 * colStep + w1];
        float gain110 = gains[y1 * rowStep + x1 * colStep + w0];
        float gain111 = gains[y1 * rowStep + x1 * colStep + w1];
        
        // Interpolate in table (w) direction.
        float gain00_ = Lerp_real32(gain000, gain001, wf);
        float gain01_ = Lerp_real32(gain010, gain011, wf);
        float gain10_ = Lerp_real32(gain100, gain101, wf);
        float gain11_ = Lerp_real32(gain110, gain111, wf);
        
        // Interpolate in column (x) direction.
        float gain0__ = Lerp_real32(gain00_, gain01_, xf);
        float gain1__ = Lerp_real32(gain10_, gain11_, xf);
        
        // Interpolate in row (y) direction.
        float gain = Lerp_real32(gain0__, gain1__, yf);
        
        // Apply gain.
        r *= gain;
        g *= gain;
        b *= gain;
        
        // Optionally clamp to [0,1].
        if (!supportOverrange) {
            r = Pin_real32(0.0f, r, 1.0f);
            g = Pin_real32(0.0f, g, 1.0f);
            b = Pin_real32(0.0f, b, 1.0f);
        }
        
        // Store the result.
        rDstPtr[col] = r;
        gDstPtr[col] = g;
        bDstPtr[col] = b;
        
        // Increment sample position for next column.
        x += 1.0f;
    }
}

// Wrapper that processes entire image using the SDK row-by-row approach
static void apply_profile_gain_table_map(
    float* rgb,  // Input/output RGB image, shape (H, W, 3), interleaved
    int height, int width,
    int points_v, int points_h,
    float spacing_v, float spacing_h,
    float origin_v, float origin_h,
    int num_table_points,
    const float* weights,
    float gamma,
    const float* gains,
    float exposure_weight_gain
) {
    // Allocate temporary planar buffers for one row
    std::vector<float> rRow(width), gRow(width), bRow(width);
    
    // Image area is the full image (0, 0, width, height)
    const int imageAreaL = 0;
    const int imageAreaT = 0;
    const int imageAreaW = width;
    const int imageAreaH = height;
    
    for (int row = 0; row < height; row++) {
        // Deinterleave this row
        for (int col = 0; col < width; col++) {
            int idx = (row * width + col) * 3;
            rRow[col] = rgb[idx + 0];
            gRow[col] = rgb[idx + 1];
            bRow[col] = rgb[idx + 2];
        }
        
        // Process using exact SDK function
        RefBaselineProfileGainTableMap(
            rRow.data(), gRow.data(), bRow.data(),  // src
            rRow.data(), gRow.data(), bRow.data(),  // dst (in-place)
            width,      // cols
            row,        // top
            0,          // left
            imageAreaL, imageAreaT, imageAreaW, imageAreaH,
            exposure_weight_gain,
            points_v, points_h,
            spacing_v, spacing_h,
            origin_v, origin_h,
            num_table_points,
            weights,
            gamma,
            gains,
            false  // supportOverrange = false for standard rendering
        );
        
        // Interleave back
        for (int col = 0; col < width; col++) {
            int idx = (row * width + col) * 3;
            rgb[idx + 0] = rRow[col];
            rgb[idx + 1] = gRow[col];
            rgb[idx + 2] = bRow[col];
        }
    }
}

// Python wrapper for ProfileGainTableMap
static PyObject* dng_color_apply_profile_gain_table_map(PyObject* self, PyObject* args) {
    PyArrayObject* rgb_array = NULL;
    PyArrayObject* gains_array = NULL;
    PyArrayObject* weights_array = NULL;
    int points_v, points_h, num_table_points;
    double spacing_v, spacing_h, origin_v, origin_h;
    double gamma;
    double baseline_exposure;
    
    if (!PyArg_ParseTuple(args, "O!O!O!iiddddidd",
            &PyArray_Type, &rgb_array,
            &PyArray_Type, &gains_array,
            &PyArray_Type, &weights_array,
            &points_v, &points_h,
            &spacing_v, &spacing_h,
            &origin_v, &origin_h,
            &num_table_points,
            &gamma,
            &baseline_exposure)) {
        return NULL;
    }
    
    // Validate input
    if (PyArray_NDIM(rgb_array) != 3 || PyArray_DIM(rgb_array, 2) != 3) {
        PyErr_SetString(PyExc_ValueError, "rgb must be shape (H, W, 3)");
        return NULL;
    }
    if (PyArray_TYPE(rgb_array) != NPY_FLOAT32) {
        PyErr_SetString(PyExc_TypeError, "rgb must be float32");
        return NULL;
    }
    if (PyArray_NDIM(gains_array) != 3) {
        PyErr_SetString(PyExc_ValueError, "gains must be 3D (points_v, points_h, num_table_points)");
        return NULL;
    }
    if (PyArray_SIZE(weights_array) != 5) {
        PyErr_SetString(PyExc_ValueError, "weights must have 5 elements");
        return NULL;
    }
    
    npy_intp height = PyArray_DIM(rgb_array, 0);
    npy_intp width = PyArray_DIM(rgb_array, 1);
    
    // Get contiguous arrays
    PyArrayObject* rgb_cont = (PyArrayObject*)PyArray_ContiguousFromAny(
        (PyObject*)rgb_array, NPY_FLOAT32, 3, 3);
    PyArrayObject* gains_cont = (PyArrayObject*)PyArray_ContiguousFromAny(
        (PyObject*)gains_array, NPY_FLOAT32, 3, 3);
    PyArrayObject* weights_cont = (PyArrayObject*)PyArray_ContiguousFromAny(
        (PyObject*)weights_array, NPY_FLOAT32, 1, 1);
    
    if (!rgb_cont || !gains_cont || !weights_cont) {
        Py_XDECREF(rgb_cont);
        Py_XDECREF(gains_cont);
        Py_XDECREF(weights_cont);
        return NULL;
    }
    
    // Copy RGB for output
    PyObject* result = PyArray_NewCopy(rgb_cont, NPY_CORDER);
    if (!result) {
        Py_DECREF(rgb_cont);
        Py_DECREF(gains_cont);
        Py_DECREF(weights_cont);
        return NULL;
    }
    
    float* result_data = (float*)PyArray_DATA((PyArrayObject*)result);
    const float* gains = (const float*)PyArray_DATA(gains_cont);
    const float* weights = (const float*)PyArray_DATA(weights_cont);
    
    float exposure_weight_gain = powf(2.0f, (float)baseline_exposure);
    
    apply_profile_gain_table_map(
        result_data, (int)height, (int)width,
        points_v, points_h,
        (float)spacing_v, (float)spacing_h,
        (float)origin_v, (float)origin_h,
        num_table_points,
        weights,
        (float)gamma,
        gains,
        exposure_weight_gain
    );
    
    Py_DECREF(rgb_cont);
    Py_DECREF(gains_cont);
    Py_DECREF(weights_cont);
    return result;
}

// Warp rectilinear lens distortion correction (Stage 2)
static PyObject* dng_color_warp_rectilinear(PyObject* self, PyObject* args, PyObject* kwargs) {
    static const char* kwlist[] = {
        "rgb", "radial_params", "center_x", "center_y", "tangential_params", "use_bicubic", NULL
    };
    
    PyArrayObject* rgb_array = NULL;
    PyArrayObject* radial_array = NULL;
    PyArrayObject* tangential_array = NULL;
    double center_x = 0.5, center_y = 0.5;
    int use_bicubic = 1;  // default True (SDK uses bicubic)
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!O!dd|O!p",
            const_cast<char**>(kwlist),
            &PyArray_Type, &rgb_array,
            &PyArray_Type, &radial_array,
            &center_x, &center_y,
            &PyArray_Type, &tangential_array,
            &use_bicubic)) {
        return NULL;
    }
    
    if (PyArray_NDIM(rgb_array) != 3 || PyArray_DIM(rgb_array, 2) != 3) {
        PyErr_SetString(PyExc_ValueError, "rgb must be shape (H, W, 3)");
        return NULL;
    }
    if (PyArray_TYPE(rgb_array) != NPY_FLOAT32) {
        PyErr_SetString(PyExc_TypeError, "rgb must be float32");
        return NULL;
    }
    
    npy_intp height = PyArray_DIM(rgb_array, 0);
    npy_intp width = PyArray_DIM(rgb_array, 1);
    
    PyArrayObject* rgb_cont = (PyArrayObject*)PyArray_ContiguousFromAny(
        (PyObject*)rgb_array, NPY_FLOAT32, 3, 3);
    // radial_array is 2D: (num_planes, num_coeffs)
    PyArrayObject* radial_cont = (PyArrayObject*)PyArray_ContiguousFromAny(
        (PyObject*)radial_array, NPY_FLOAT64, 2, 2);
    PyArrayObject* tan_cont = NULL;
    
    if (tangential_array) {
        // tangential_array is 2D: (num_planes, 2)
        tan_cont = (PyArrayObject*)PyArray_ContiguousFromAny(
            (PyObject*)tangential_array, NPY_FLOAT64, 2, 2);
    }
    
    if (!rgb_cont || !radial_cont) {
        Py_XDECREF(rgb_cont);
        Py_XDECREF(radial_cont);
        Py_XDECREF(tan_cont);
        return NULL;
    }
    
    // Get dimensions from 2D radial array
    int num_planes = (int)PyArray_DIM(radial_cont, 0);
    int num_coeffs = (int)PyArray_DIM(radial_cont, 1);
    
    npy_intp dims[3] = {height, width, 3};
    PyObject* result = PyArray_SimpleNew(3, dims, NPY_FLOAT32);
    if (!result) {
        Py_DECREF(rgb_cont);
        Py_DECREF(radial_cont);
        Py_XDECREF(tan_cont);
        return NULL;
    }
    
    const float* src = (const float*)PyArray_DATA(rgb_cont);
    float* dst = (float*)PyArray_DATA((PyArrayObject*)result);
    const double* radial = (const double*)PyArray_DATA(radial_cont);
    const double* tangential = tan_cont ? (const double*)PyArray_DATA(tan_cont) : NULL;
    
    warp_rectilinear(src, dst, height, width, 3, radial, num_planes, num_coeffs, tangential, center_x, center_y, (bool)use_bicubic);
    
    Py_DECREF(rgb_cont);
    Py_DECREF(radial_cont);
    Py_XDECREF(tan_cont);
    return result;
}

// Fix vignette radial (Stage 2)
static PyObject* dng_color_fix_vignette(PyObject* self, PyObject* args) {
    PyArrayObject* rgb_array = NULL;
    PyArrayObject* params_array = NULL;
    double center_x = 0.5, center_y = 0.5;
    
    if (!PyArg_ParseTuple(args, "O!O!dd",
            &PyArray_Type, &rgb_array,
            &PyArray_Type, &params_array,
            &center_x, &center_y)) {
        return NULL;
    }
    
    if (PyArray_NDIM(rgb_array) != 3 || PyArray_DIM(rgb_array, 2) != 3) {
        PyErr_SetString(PyExc_ValueError, "rgb must be shape (H, W, 3)");
        return NULL;
    }
    if (PyArray_TYPE(rgb_array) != NPY_FLOAT32) {
        PyErr_SetString(PyExc_TypeError, "rgb must be float32");
        return NULL;
    }
    
    npy_intp height = PyArray_DIM(rgb_array, 0);
    npy_intp width = PyArray_DIM(rgb_array, 1);
    
    PyArrayObject* rgb_cont = (PyArrayObject*)PyArray_ContiguousFromAny(
        (PyObject*)rgb_array, NPY_FLOAT32, 3, 3);
    PyArrayObject* params_cont = (PyArrayObject*)PyArray_ContiguousFromAny(
        (PyObject*)params_array, NPY_FLOAT64, 1, 1);
    
    if (!rgb_cont || !params_cont) {
        Py_XDECREF(rgb_cont);
        Py_XDECREF(params_cont);
        return NULL;
    }
    
    // Copy for output
    PyObject* result = PyArray_NewCopy(rgb_cont, NPY_CORDER);
    if (!result) {
        Py_DECREF(rgb_cont);
        Py_DECREF(params_cont);
        return NULL;
    }
    
    float* data = (float*)PyArray_DATA((PyArrayObject*)result);
    const double* params = (const double*)PyArray_DATA(params_cont);
    int num_params = (int)PyArray_SIZE(params_cont);
    
    fix_vignette_radial(data, height, width, 3, params, num_params, center_x, center_y);
    
    Py_DECREF(rgb_cont);
    Py_DECREF(params_cont);
    return result;
}

//=============================================================================
// Bilinear Demosaic (from dng_mosaic_info.cpp dng_bilinear_interpolator)
//
// This is a port of the DNG SDK's bilinear demosaicing algorithm.
// For a 2x2 Bayer CFA pattern, each output pixel needs R, G, B values.
// At positions where that color exists in the CFA, copy directly.
// At other positions, interpolate from neighbors using bilinear weights.
//
// SDK reference: dng_mosaic_info.cpp lines 31-1090, dng_reference.cpp
// RefBilinearRow16/RefBilinearRow32 lines 1293-1385
//=============================================================================

// CFA pattern is passed as a 2x2 array of color plane indices
// SDK ref: dng_mosaic_info::fCFAPattern[row][col] contains plane index
// Color plane indices: 0=Red, 1=Green, 2=Blue (per fCFAPlaneColor)

// Get CFA color at position (row, col) for 2x2 pattern
static inline int get_cfa_color(const int* cfa_colors, int row, int col) {
    int pr = row & 1;
    int pc = col & 1;
    return cfa_colors[pr * 2 + pc];
}

// Bilinear demosaic for 2x2 Bayer pattern
// SDK ref: dng_mosaic_info.cpp dng_bilinear_pattern::Calculate() and
//          dng_bilinear_interpolator::Interpolate()
//
// For each output color plane, the kernel weights for a 2x2 Bayer pattern are:
//
// Red plane (planeColor=0):
//   At R position: copy (weight 1.0)
//   At G position in R row: average of E and W red neighbors
//   At G position in B row: average of N and S red neighbors  
//   At B position: average of 4 diagonal red neighbors (NW, NE, SW, SE)
//
// Blue plane (planeColor=2): same logic but for blue
//
// Green plane (planeColor=1):
//   At G position: copy (weight 1.0)
//   At R or B position: average of 4 adjacent green neighbors (N, S, E, W)
//
// SDK operates on float32 throughout (dng_pixel_buffer with ttFloat)
//
static void bilinear_demosaic_kernel(
    const float* src, float* dst,
    npy_intp height, npy_intp width,
    const int* cfa_colors
) {
    // Process each output pixel
    for (npy_intp row = 0; row < height; row++) {
        for (npy_intp col = 0; col < width; col++) {
            npy_intp dst_idx = (row * width + col) * 3;
            int this_color = get_cfa_color(cfa_colors, row, col);
            
            // For each output color plane (R=0, G=1, B=2)
            for (int plane = 0; plane < 3; plane++) {
                float sum = 0.0f;
                int count = 0;
                
                if (this_color == plane) {
                    // This position has the color we want - copy directly
                    // SDK ref: lines 606-614 "Special case no interpolation case"
                    dst[dst_idx + plane] = src[row * width + col];
                }
                else if (plane == 1) {
                    // Green interpolation at R or B position
                    // SDK ref: lines 634-646 "All sides" - average N,S,E,W
                    // Check N
                    if (row > 0) {
                        sum += src[(row - 1) * width + col];
                        count++;
                    }
                    // Check S
                    if (row < height - 1) {
                        sum += src[(row + 1) * width + col];
                        count++;
                    }
                    // Check W
                    if (col > 0) {
                        sum += src[row * width + (col - 1)];
                        count++;
                    }
                    // Check E
                    if (col < width - 1) {
                        sum += src[row * width + (col + 1)];
                        count++;
                    }
                    dst[dst_idx + plane] = sum / (float)count;
                }
                else {
                    // Red or Blue interpolation
                    // Determine if we're interpolating R at B position or B at R position
                    // or R/B at G position
                    
                    // Find which color is at this position
                    if (this_color == 1) {
                        // Green position - need to interpolate R or B
                        // SDK ref: lines 648-670 "N & S" or "E & W"
                        // Determine if horizontal or vertical neighbors have this color
                        
                        // Check if horizontal neighbors (E,W) have the target color
                        int west_color = (col > 0) ? get_cfa_color(cfa_colors, row, col - 1) : -1;
                        int east_color = (col < width - 1) ? get_cfa_color(cfa_colors, row, col + 1) : -1;
                        
                        if (west_color == plane || east_color == plane) {
                            // Horizontal interpolation
                            if (col > 0 && get_cfa_color(cfa_colors, row, col - 1) == plane) {
                                sum += src[row * width + (col - 1)];
                                count++;
                            }
                            if (col < width - 1 && get_cfa_color(cfa_colors, row, col + 1) == plane) {
                                sum += src[row * width + (col + 1)];
                                count++;
                            }
                        } else {
                            // Vertical interpolation
                            if (row > 0 && get_cfa_color(cfa_colors, row - 1, col) == plane) {
                                sum += src[(row - 1) * width + col];
                                count++;
                            }
                            if (row < height - 1 && get_cfa_color(cfa_colors, row + 1, col) == plane) {
                                sum += src[(row + 1) * width + col];
                                count++;
                            }
                        }
                        if (count > 0) {
                            dst[dst_idx + plane] = sum / (float)count;
                        } else {
                            dst[dst_idx + plane] = 0.0f;
                        }
                    }
                    else {
                        // R position needing B, or B position needing R
                        // SDK ref: lines 724-736 "Four corners" - diagonal average
                        if (row > 0 && col > 0) {
                            sum += src[(row - 1) * width + (col - 1)];
                            count++;
                        }
                        if (row > 0 && col < width - 1) {
                            sum += src[(row - 1) * width + (col + 1)];
                            count++;
                        }
                        if (row < height - 1 && col > 0) {
                            sum += src[(row + 1) * width + (col - 1)];
                            count++;
                        }
                        if (row < height - 1 && col < width - 1) {
                            sum += src[(row + 1) * width + (col + 1)];
                            count++;
                        }
                        if (count > 0) {
                            dst[dst_idx + plane] = sum / (float)count;
                        } else {
                            dst[dst_idx + plane] = 0.0f;
                        }
                    }
                }
            }
        }
    }
}

// Python wrapper for bilinear demosaic
// SDK ref: dng_mosaic_info::InterpolateGeneric
//
// Args:
//   cfa: Normalized CFA data, float32, shape (H, W), values in [0,1]
//   cfa_pattern: 2x2 array of color plane indices [row0col0, row0col1, row1col0, row1col1]
//                SDK ref: dng_mosaic_info::fCFAPattern[row][col]
//                Values: 0=Red, 1=Green, 2=Blue (per fCFAPlaneColor)
//
static PyObject* dng_color_bilinear_demosaic(PyObject* self, PyObject* args, PyObject* kwargs) {
    PyArrayObject* cfa_array = NULL;
    PyArrayObject* cfa_pattern_array = NULL;
    
    static char* kwlist[] = {(char*)"cfa", (char*)"cfa_pattern", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!O!", kwlist,
            &PyArray_Type, &cfa_array,
            &PyArray_Type, &cfa_pattern_array)) {
        return NULL;
    }
    
    // Validate CFA pattern array - must be 4 elements (2x2 flattened)
    if (PyArray_SIZE(cfa_pattern_array) != 4) {
        PyErr_SetString(PyExc_ValueError, 
            "cfa_pattern must be 4 elements (2x2 pattern flattened row-major)");
        return NULL;
    }
    
    // Validate CFA array
    if (PyArray_NDIM(cfa_array) != 2) {
        PyErr_SetString(PyExc_ValueError, "cfa must be 2D array (H, W)");
        return NULL;
    }
    if (PyArray_TYPE(cfa_array) != NPY_FLOAT32) {
        PyErr_SetString(PyExc_TypeError, "cfa must be float32");
        return NULL;
    }
    
    npy_intp height = PyArray_DIM(cfa_array, 0);
    npy_intp width = PyArray_DIM(cfa_array, 1);
    
    if (height < 2 || width < 2) {
        PyErr_SetString(PyExc_ValueError, "Image must be at least 2x2");
        return NULL;
    }
    
    // Ensure contiguous input
    PyArrayObject* cfa_cont = (PyArrayObject*)PyArray_ContiguousFromAny(
        (PyObject*)cfa_array, NPY_FLOAT32, 2, 2);
    if (!cfa_cont) return NULL;
    
    // Get CFA pattern as int array
    PyArrayObject* pattern_cont = (PyArrayObject*)PyArray_ContiguousFromAny(
        (PyObject*)cfa_pattern_array, NPY_INT32, 1, 2);
    if (!pattern_cont) {
        Py_DECREF(cfa_cont);
        return NULL;
    }
    
    const int32_t* pattern_data = (const int32_t*)PyArray_DATA(pattern_cont);
    int cfa_colors[4] = {pattern_data[0], pattern_data[1], pattern_data[2], pattern_data[3]};
    
    // Validate color indices (must be 0, 1, or 2)
    for (int i = 0; i < 4; i++) {
        if (cfa_colors[i] < 0 || cfa_colors[i] > 2) {
            Py_DECREF(cfa_cont);
            Py_DECREF(pattern_cont);
            PyErr_SetString(PyExc_ValueError, 
                "cfa_pattern values must be 0 (Red), 1 (Green), or 2 (Blue)");
            return NULL;
        }
    }
    
    // Create output array (float32, same as SDK)
    npy_intp dims[3] = {height, width, 3};
    PyObject* result = PyArray_SimpleNew(3, dims, NPY_FLOAT32);
    if (!result) {
        Py_DECREF(cfa_cont);
        Py_DECREF(pattern_cont);
        return NULL;
    }
    
    const float* src_data = (const float*)PyArray_DATA(cfa_cont);
    float* dst_data = (float*)PyArray_DATA((PyArrayObject*)result);
    
    // Run bilinear demosaic
    bilinear_demosaic_kernel(src_data, dst_data, height, width, cfa_colors);
    
    Py_DECREF(cfa_cont);
    Py_DECREF(pattern_cont);
    return result;
}

// =============================================================================
// MapPolynomial opcode (OpcodeList2)
// SDK ref: dng_reference.cpp RefBaselineMapPoly32
// =============================================================================

// Port of RefBaselineMapPoly32 from dng_reference.cpp
// Note: Uses existing Pin_real32() defined above
// Applies polynomial: y = c0 + c1*x + c2*x^2 + ... 
// For negative x, alternates signs on even powers
static void RefBaselineMapPoly32(
    float* dPtr,
    const int32_t rowStep,
    const uint32_t rows,
    const uint32_t cols,
    const uint32_t rowPitch,
    const uint32_t colPitch,
    const float* coefficients,
    const uint32_t degree,
    uint16_t blackLevel)
{
    float blackScale1 = 1.0f;
    float blackScale2 = 1.0f;
    float blackOffset1 = 0.0f;
    float blackOffset2 = 0.0f;

    if (blackLevel != 0) {
        blackOffset2 = ((float)blackLevel) / 65535.0f;
        blackScale2 = 1.0f - blackOffset2;
        blackScale1 = (blackScale2 != 0.0f) ? 1.0f / blackScale2 : 0.0f;
        blackOffset1 = 1.0f - blackScale1;
    }

    for (uint32_t row = 0; row < rows; row += rowPitch) {
        
        if (blackLevel != 0) {
            for (uint32_t col = 0; col < cols; col += colPitch) {
                dPtr[col] = dPtr[col] * blackScale1 + blackOffset1;
            }
        }

        switch (degree) {
            case 0: {
                float y = Pin_real32(-1.0f, coefficients[0], 1.0f);
                for (uint32_t col = 0; col < cols; col += colPitch) {
                    dPtr[col] = y;
                }
                break;
            }

            case 1: {
                for (uint32_t col = 0; col < cols; col += colPitch) {
                    float x = dPtr[col];
                    float y = coefficients[0] + x * coefficients[1];
                    dPtr[col] = Pin_real32(-1.0f, y, 1.0f);
                }
                break;
            }

            case 2: {
                for (uint32_t col = 0; col < cols; col += colPitch) {
                    float x = dPtr[col];
                    float y;
                    if (x < 0.0f) {
                        y = coefficients[0] + x * (coefficients[1] - x * coefficients[2]);
                    } else {
                        y = coefficients[0] + x * (coefficients[1] + x * coefficients[2]);
                    }
                    dPtr[col] = Pin_real32(-1.0f, y, 1.0f);
                }
                break;
            }

            case 3: {
                for (uint32_t col = 0; col < cols; col += colPitch) {
                    float x = dPtr[col];
                    float y;
                    if (x < 0.0f) {
                        y = coefficients[0] + x * (coefficients[1] - x * (coefficients[2] - x * coefficients[3]));
                    } else {
                        y = coefficients[0] + x * (coefficients[1] + x * (coefficients[2] + x * coefficients[3]));
                    }
                    dPtr[col] = Pin_real32(-1.0f, y, 1.0f);
                }
                break;
            }

            case 4: {
                for (uint32_t col = 0; col < cols; col += colPitch) {
                    float x = dPtr[col];
                    float y;
                    if (x < 0.0f) {
                        y = coefficients[0] + x * (coefficients[1] - x * (coefficients[2] - x * (coefficients[3] - x * coefficients[4])));
                    } else {
                        y = coefficients[0] + x * (coefficients[1] + x * (coefficients[2] + x * (coefficients[3] + x * coefficients[4])));
                    }
                    dPtr[col] = Pin_real32(-1.0f, y, 1.0f);
                }
                break;
            }

            default: {
                for (uint32_t col = 0; col < cols; col += colPitch) {
                    float x = dPtr[col];
                    float y = coefficients[0];

                    if (x < 0.0f) {
                        x = -x;
                        float xx = x;
                        for (uint32_t j = 1; j <= degree; j++) {
                            y -= coefficients[j] * xx;
                            xx *= x;
                        }
                    } else {
                        float xx = x;
                        for (uint32_t j = 1; j <= degree; j++) {
                            y += coefficients[j] * xx;
                            xx *= x;
                        }
                    }
                    dPtr[col] = Pin_real32(-1.0f, y, 1.0f);
                }
                break;
            }
        }

        if (blackLevel != 0) {
            for (uint32_t col = 0; col < cols; col += colPitch) {
                dPtr[col] = dPtr[col] * blackScale2 + blackOffset2;
            }
        }

        dPtr += rowStep;
    }
}

// Python wrapper for apply_map_polynomial
static PyObject* dng_color_apply_map_polynomial(PyObject* self, PyObject* args) {
    PyObject* rgb_obj;
    PyObject* coeffs_obj;
    int top, left, bottom, right;
    int plane, planes, row_pitch, col_pitch;
    int degree;
    
    if (!PyArg_ParseTuple(args, "OOiiiiiiiii",
            &rgb_obj, &coeffs_obj,
            &top, &left, &bottom, &right,
            &plane, &planes, &row_pitch, &col_pitch,
            &degree)) {
        return NULL;
    }
    
    // Get contiguous arrays
    PyObject* rgb_cont = PyArray_ContiguousFromAny(rgb_obj, NPY_FLOAT32, 3, 3);
    PyObject* coeffs_cont = PyArray_ContiguousFromAny(coeffs_obj, NPY_FLOAT32, 1, 1);
    if (!rgb_cont || !coeffs_cont) {
        Py_XDECREF(rgb_cont);
        Py_XDECREF(coeffs_cont);
        return NULL;
    }
    
    npy_intp* dims = PyArray_DIMS((PyArrayObject*)rgb_cont);
    int height = (int)dims[0];
    int width = (int)dims[1];
    int num_planes = (int)dims[2];
    
    // Handle area bounds (0 means full image per SDK)
    if (top == 0 && bottom == 0) { top = 0; bottom = height; }
    if (left == 0 && right == 0) { left = 0; right = width; }
    
    // Create output (copy input)
    PyObject* result = PyArray_Copy((PyArrayObject*)rgb_cont);
    if (!result) {
        Py_DECREF(rgb_cont);
        Py_DECREF(coeffs_cont);
        return NULL;
    }
    
    float* dst_data = (float*)PyArray_DATA((PyArrayObject*)result);
    const float* coefficients = (const float*)PyArray_DATA((PyArrayObject*)coeffs_cont);
    
    // Apply to each specified plane
    // For interleaved RGB (H,W,3), each row has width*3 elements
    // SDK iterates: for (col = 0; col < cols; col += colPitch)
    // So cols must be width * num_planes, and colPitch = num_planes to hit same plane
    int area_height = bottom - top;
    int area_width = right - left;
    
    for (int p = plane; p < plane + planes && p < num_planes; p++) {
        // Get pointer to start of area for this plane
        float* plane_ptr = dst_data + (top * width + left) * num_planes + p;
        
        // rowStep: elements to advance per row (width * num_planes)
        int32_t rowStep = (int32_t)(width * num_planes * row_pitch);
        
        // cols: total elements in row (area_width * num_planes)
        // colPitch: skip between same-plane values (num_planes * col_pitch)
        RefBaselineMapPoly32(
            plane_ptr,
            rowStep,
            (uint32_t)area_height,
            (uint32_t)(area_width * num_planes),  // cols = total elements in row
            (uint32_t)row_pitch,
            (uint32_t)(col_pitch * num_planes),   // colPitch to skip between same-plane
            coefficients,
            (uint32_t)degree,
            0  // blackLevel - stage 2 is already normalized
        );
    }
    
    Py_DECREF(rgb_cont);
    Py_DECREF(coeffs_cont);
    return result;
}

// Module method definitions
static PyMethodDef DngColorMethods[] = {
    {"temp_to_xy", dng_color_temp_to_xy, METH_VARARGS,
     "Convert color temperature and tint to xy chromaticity coordinates.\n\n"
     "Args:\n"
     "    temperature (float): Color temperature in Kelvin (1667-25000)\n"
     "    tint (float): Green/magenta tint (-150 to +150)\n\n"
     "Returns:\n"
     "    tuple: (x, y) chromaticity coordinates"},
    
    {"xy_to_temp", dng_color_xy_to_temp, METH_VARARGS,
     "Convert xy chromaticity coordinates to color temperature and tint.\n\n"
     "Args:\n"
     "    x (float): x chromaticity coordinate\n"
     "    y (float): y chromaticity coordinate\n\n"
     "Returns:\n"
     "    tuple: (temperature, tint)"},
    
    {"get_acr3_curve", dng_color_get_acr3_curve, METH_VARARGS,
     "Get the ACR3 default tone curve as a lookup table.\n\n"
     "Args:\n"
     "    num_points (int): Number of points in the LUT (default: 256)\n\n"
     "Returns:\n"
     "    ndarray: 1D array of tone curve values"},
    
    {"apply_hue_sat_map", dng_color_apply_hue_sat_map, METH_VARARGS,
     "Apply HueSatMap (3D LUT) to RGB image for camera profile color adjustments.\n\n"
     "Args:\n"
     "    rgb (ndarray): Input RGB image, float32, shape (H, W, 3)\n"
     "    hue_sat_map (ndarray): HueSatMap data, float32, shape (V, H, S, 3) or flattened\n"
     "        Each entry contains (hue_shift_degrees, sat_scale, val_scale)\n"
     "    hue_divs (int): Number of hue divisions in the map\n"
     "    sat_divs (int): Number of saturation divisions in the map\n"
     "    val_divs (int): Number of value divisions in the map\n\n"
     "Returns:\n"
     "    ndarray: Processed RGB image with HueSatMap adjustments applied"},
    
    {"interpolate_matrices", dng_color_interpolate_matrices, METH_VARARGS,
     "Interpolate dual-illuminant matrices by color temperature.\n\n"
     "Uses inverse temperature (1/T) interpolation as per DNG spec.\n\n"
     "Args:\n"
     "    matrix1 (ndarray): 3x3 matrix for illuminant 1 (e.g., ColorMatrix1)\n"
     "    matrix2 (ndarray): 3x3 matrix for illuminant 2 (e.g., ColorMatrix2)\n"
     "    temp1 (float): Color temperature of illuminant 1 (Kelvin)\n"
     "    temp2 (float): Color temperature of illuminant 2 (Kelvin)\n"
     "    target_temp (float): Target color temperature to interpolate for\n\n"
     "Returns:\n"
     "    ndarray: Interpolated 3x3 matrix"},
    
    {"apply_tone_curve", dng_color_apply_tone_curve, METH_VARARGS,
     "Apply a custom tone curve (ProfileToneCurve) to RGB image.\n\n"
     "Args:\n"
     "    rgb (ndarray): Input RGB image, float32, shape (H, W, 3)\n"
     "    curve (ndarray): 1D tone curve LUT, float32, maps [0,1] -> [0,1]\n\n"
     "Returns:\n"
     "    ndarray: Tone-mapped RGB image"},
    
    {"apply_rgb_tone", dng_color_apply_rgb_tone, METH_VARARGS,
     "Apply hue-preserving RGB tone curve (RefBaselineRGBTone).\n\n"
     "This is the SDK's default tone mapping that preserves color relationships\n"
     "by applying the curve to max/min channels and interpolating the middle.\n\n"
     "Args:\n"
     "    rgb (ndarray): Input RGB image, float32, shape (H, W, 3)\n"
     "    curve (ndarray): 1D tone curve LUT, float32, maps [0,1] -> [0,1]\n\n"
     "Returns:\n"
     "    ndarray: Hue-preserving tone-mapped RGB image"},
    
    {"apply_exposure_ramp", dng_color_apply_exposure_ramp, METH_VARARGS,
     "Apply exposure ramp function (dng_function_exposure_ramp).\n\n"
     "SDK ref: dng_render.cpp lines 50-103\n"
     "3 regions: below black-radius=0, above black+radius=linear, between=quadratic\n\n"
     "Args:\n"
     "    rgb (ndarray): Input RGB image, float32, shape (H, W, 3)\n"
     "    white (float): White point (1.0 / pow(2, max(0, exposure)))\n"
     "    black (float): Black point (shadows * shadowScale * 0.001)\n"
     "    minBlack (float): Minimum black for radius calculation\n"
     "    supportOverrange (bool, optional): Allow values > 1.0\n\n"
     "Returns:\n"
     "    ndarray: Exposure-adjusted RGB image"},
    
    {"apply_exposure_tone", dng_color_apply_exposure_tone, METH_VARARGS,
     "Apply exposure tone function (dng_function_exposure_tone).\n\n"
     "SDK ref: dng_render.cpp lines 107-157\n"
     "For negative exposure: darkens using piecewise linear+quadratic.\n"
     "For non-negative exposure: returns input unchanged (NOP).\n\n"
     "Args:\n"
     "    rgb (ndarray): Input RGB image, float32, shape (H, W, 3)\n"
     "    exposure (float): Total exposure value (BaselineExposure + offset)\n\n"
     "Returns:\n"
     "    ndarray: Exposure-tone-adjusted RGB image"},
    
    {"srgb_gamma", dng_color_srgb_gamma, METH_VARARGS,
     "Apply sRGB gamma encoding (linear to sRGB).\n\n"
     "Args:\n"
     "    rgb (ndarray): Input linear RGB image, float32, shape (H, W, 3)\n\n"
     "Returns:\n"
     "    ndarray: sRGB gamma-encoded image"},
    
    {"matrix_transform", (PyCFunction)dng_color_matrix_transform, METH_VARARGS | METH_KEYWORDS,
     "Apply 3x3 color matrix transform to RGB image.\n\n"
     "Args:\n"
     "    rgb (ndarray): Input RGB image, float32, shape (H, W, 3)\n"
     "    matrix (ndarray): 3x3 color transform matrix, float32\n"
     "    clip (bool): Clip output to [0,1] (default: True)\n\n"
     "Returns:\n"
     "    ndarray: Transformed RGB image"},
    
    {"bradford_adapt", dng_color_bradford_adapt, METH_VARARGS,
     "Compute Bradford chromatic adaptation matrix between two white points.\n\n"
     "Args:\n"
     "    src_x (float): Source white point x chromaticity\n"
     "    src_y (float): Source white point y chromaticity\n"
     "    dst_x (float): Destination white point x chromaticity\n"
     "    dst_y (float): Destination white point y chromaticity\n\n"
     "Returns:\n"
     "    ndarray: 3x3 chromatic adaptation matrix"},
    
    {"linearize", dng_color_linearize, METH_VARARGS,
     "Apply linearization table to RAW sensor data (Stage 1).\n\n"
     "Converts non-linear sensor ADC values to linear light values.\n\n"
     "Args:\n"
     "    data (ndarray): RAW sensor data, float32\n"
     "    table (ndarray): Linearization LUT, float32\n"
     "    max_val (float): Maximum input value (e.g., 16383 for 14-bit)\n\n"
     "Returns:\n"
     "    ndarray: Linearized data"},
    
    {"normalize_raw", (PyCFunction)dng_color_normalize_raw, METH_VARARGS | METH_KEYWORDS,
     "Normalize RAW data using black and white levels per DNG spec Chapter 5.\n\n"
     "Implements: linear = (raw - BlackLevel[r%rR][c%rC][s] - DeltaH[c] - DeltaV[r]) / (WhiteLevel[s] - BlackLevel)\n\n"
     "Args:\n"
     "    data (ndarray): RAW pixel data, float32, (H,W) or (H,W,samples_per_pixel)\n"
     "    black_level (ndarray): BlackLevel pattern, float32, flattened in row-col-sample order\n"
     "    black_repeat_rows (int): Number of rows in repeating pattern (from BlackLevelRepeatDim[0])\n"
     "    black_repeat_cols (int): Number of cols in repeating pattern (from BlackLevelRepeatDim[1])\n"
     "    samples_per_pixel (int): 1 for CFA, 3 for LinearRaw\n"
     "    white_level (ndarray): WhiteLevel per sample, float32\n"
     "    black_delta_h (ndarray, optional): Per-column delta, float32, shape (width,)\n"
     "    black_delta_v (ndarray, optional): Per-row delta, float32, shape (height,)\n\n"
     "Returns:\n"
     "    ndarray: Normalized data in [0,1] range"},
    
    {"apply_gain_map", dng_color_apply_gain_map, METH_VARARGS,
     "Apply gain map (flat-field correction) to RAW CFA data (Stage 1).\n\n"
     "Multiplies each CFA pixel by interpolated gain value.\n\n"
     "Args:\n"
     "    data (ndarray): RAW CFA data, float32, (H,W)\n"
     "    gain_map (ndarray): 2D gain map, float32\n\n"
     "Returns:\n"
     "    ndarray: Gain-corrected CFA data"},
    
    {"apply_profile_gain_table_map", dng_color_apply_profile_gain_table_map, METH_VARARGS,
     "Apply ProfileGainTableMap to RGB image (Stage 3).\n\n"
     "SDK ref: dng_reference.cpp RefBaselineProfileGainTableMap()\n"
     "Applied after HueSatMap, before exposure ramp.\n\n"
     "Args:\n"
     "    rgb (ndarray): RGB image, float32, (H,W,3)\n"
     "    gains (ndarray): Gain table, float32, (points_v, points_h, num_table_points)\n"
     "    weights (ndarray): MapInputWeights, float32, (5,)\n"
     "    points_v (int): Map height\n"
     "    points_h (int): Map width\n"
     "    spacing_v (float): Vertical spacing\n"
     "    spacing_h (float): Horizontal spacing\n"
     "    origin_v (float): Vertical origin\n"
     "    origin_h (float): Horizontal origin\n"
     "    num_table_points (int): Table depth\n"
     "    gamma (float): Gamma parameter\n"
     "    baseline_exposure (float): BaselineExposure value\n\n"
     "Returns:\n"
     "    ndarray: RGB with gain applied"},
    
    {"warp_rectilinear", (PyCFunction)dng_color_warp_rectilinear, METH_VARARGS | METH_KEYWORDS,
     "Apply lens distortion correction using WarpRectilinear opcode (Stage 2).\n\n"
     "Uses polynomial radial model: r_src = r_dst * f(r_dst)\n"
     "where f(r) = k0 + k1*r + k2*r^2 + k3*r^3\n\n"
     "Args:\n"
     "    rgb (ndarray): Input RGB image, float32, (H,W,3)\n"
     "    radial_params (ndarray): Radial polynomial coefficients [k0,k1,k2,k3]\n"
     "    center_x (float): Optical center x in [0,1] (default: 0.5)\n"
     "    center_y (float): Optical center y in [0,1] (default: 0.5)\n"
     "    tangential_params (ndarray): Optional tangential coefficients [kt0,kt1]\n\n"
     "Returns:\n"
     "    ndarray: Distortion-corrected RGB image"},
    
    {"fix_vignette", dng_color_fix_vignette, METH_VARARGS,
     "Apply radial vignette correction (Stage 2).\n\n"
     "Applies gain = 1 + k0*r^2 + k1*r^4 + k2*r^6 + ...\n\n"
     "Args:\n"
     "    rgb (ndarray): Input RGB image, float32, (H,W,3)\n"
     "    params (ndarray): Vignette polynomial coefficients [k0,k1,k2,...]\n"
     "    center_x (float): Optical center x in [0,1]\n"
     "    center_y (float): Optical center y in [0,1]\n\n"
     "Returns:\n"
     "    ndarray: Vignette-corrected RGB image"},
    
    {"bilinear_demosaic", (PyCFunction)dng_color_bilinear_demosaic, METH_VARARGS | METH_KEYWORDS,
     "Demosaic CFA data using DNG SDK bilinear interpolation.\n\n"
     "This is a port of the DNG SDK's bilinear demosaicing algorithm from\n"
     "dng_mosaic_info.cpp (InterpolateGeneric with dng_bilinear_interpolator).\n"
     "SDK operates on float32 throughout.\n\n"
     "Args:\n"
     "    cfa (ndarray): Normalized CFA data, float32, shape (H, W), values in [0,1]\n"
     "    cfa_pattern (ndarray): 2x2 CFA pattern as color plane indices, int32,\n"
     "        flattened row-major [row0col0, row0col1, row1col0, row1col1].\n"
     "        SDK ref: dng_mosaic_info::fCFAPattern[row][col]\n"
     "        Values: 0=Red, 1=Green, 2=Blue (per fCFAPlaneColor)\n\n"
     "Returns:\n"
     "    ndarray: Demosaiced RGB image, float32, shape (H, W, 3)"},
    
    {"apply_map_polynomial", dng_color_apply_map_polynomial, METH_VARARGS,
     "Apply MapPolynomial opcode to RGB image (OpcodeList2 Stage 2).\n\n"
     "SDK ref: dng_reference.cpp RefBaselineMapPoly32\n"
     "Applies polynomial: y = c0 + c1*x + c2*x^2 + ...\n"
     "For negative x, alternates signs on even powers.\n\n"
     "Args:\n"
     "    rgb (ndarray): RGB image, float32, (H,W,3)\n"
     "    coefficients (ndarray): Polynomial coefficients, float32\n"
     "    top, left, bottom, right (int): Area bounds (0 = full)\n"
     "    plane (int): First plane to process\n"
     "    planes (int): Number of planes\n"
     "    row_pitch (int): Row stride\n"
     "    col_pitch (int): Column stride\n"
     "    degree (int): Polynomial degree\n\n"
     "Returns:\n"
     "    ndarray: RGB with polynomial applied"},
    
    {NULL, NULL, 0, NULL}
};

// Module definition
static struct PyModuleDef dng_color_module = {
    PyModuleDef_HEAD_INIT,
    "_dng_color",
    "DNG SDK color processing (standalone implementation).\n\n"
    "This module provides access to Adobe DNG SDK's color processing algorithms,\n"
    "including color temperature conversion, camera color space transforms,\n"
    "and the ACR3 default tone curve.\n\n"
    "Based on Adobe DNG SDK 1.7.1",
    -1,
    DngColorMethods
};

// Module initialization
PyMODINIT_FUNC PyInit__dng_color(void) {
    import_array();
    return PyModule_Create(&dng_color_module);
}
