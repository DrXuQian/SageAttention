#pragma once
struct acFloatComplex { float x, y; };
struct acDoubleComplex { double x, y; };
inline acFloatComplex make_acFloatComplex(float r, float i) { return {r, i}; }
inline acDoubleComplex make_acDoubleComplex(double r, double i) { return {r, i}; }
inline float acCrealf(acFloatComplex z) { return z.x; }
inline float acCimagf(acFloatComplex z) { return z.y; }
inline double acCreal(acDoubleComplex z) { return z.x; }
inline double acCimag(acDoubleComplex z) { return z.y; }
