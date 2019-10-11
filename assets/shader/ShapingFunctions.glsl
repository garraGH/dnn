#type vertex
#version 460 core
layout(location = 0) in vec3 a_Position;
uniform mat4 u_ViewProjection;
uniform mat4 u_Transform;

void main()
{
    gl_Position = u_ViewProjection*u_Transform*vec4(a_Position, 1.0f);
}

#type fragment
#version 460 core
uniform vec4 u_Color;
uniform vec2 u_Resolution;
// uniform float u_Time;
uniform int u_ShapingFunction;

out vec4 color;

float Linear(float x)
{
    return x;
}

float Step(float x, float threshold)
{
    return step(threshold, x);
}

float SmoothStep(float x, float beg, float end)
{
    return smoothstep(beg, end, x);
}

float Power(float x, float p)
{
    return pow(x, p);
}

#define PI 3.141593
float Sine(float x)
{
    x *= 2*PI;
    return (1+sin(x))*0.5;
}

float Cosine(float x)
{
    x *= 2*PI;
    return (1+cos(x))*0.5;
}

float BlinnWyvillCosineApproximation(float x)
{
    float x2 = x*x;
    float x4 = x2*x2;
    float x6 = x2*x4;
    return (4*x6-17*x4+22*x2)/9.0;
}

vec2 Clamp(float a, float b)
{
    float epsilon = 0.00001;
    float min_a = 0.0+epsilon;
    float max_a = 1.0-epsilon;
    float min_b = 0.0;
    float max_b = 1.0;
    a = min(max_a, max(min_a, a));
    b = min(max_b, max(min_b, b));
    vec2 f = vec2(a, b);
    return f;
}

float DoubleCubicSeat(float x, float a, float b)
{
    vec2 f = Clamp(a, b);
    return x <= f.x? (f.y-f.y*pow(1-x/f.x, 3.0)) : (f.y+(1-f.y)*pow((x-f.x)/(1-f.x), 3));
}

float DoubleCubicSeatWidthLinearBlend(float x, float a, float b)
{
    vec2 f = Clamp(a, b);
    f.y = 1.0-f.y;
    return x <= f.x? (f.y*x+(1-f.y)*f.x*(1-pow(1-x/f.x, 3.0))) : (f.y*x+(1-f.y)*(f.x+(1-f.x)*pow((x-f.x)/(1-f.x), 3.0)));
}

float DoubleOddPolynomialSeat(float x, float a, float b, int n)
{
    vec2 f = Clamp(a, b);
    int p = 2*n+1;
    return x <= a? f.y-f.y*pow(1-x/f.x, p) : f.y+(1-f.y)*pow((x-f.x)/(1-f.x), p);
}

float SymmetricDoublePolynomialSigmoids(float x, float a, float b, int n)
{
    if(n%2 == 0)
    {
        return x <= 0.5? pow(2*x, 2*n)/2.0 : 1.0-pow(2*(x-1), 2*n)/2.0;
    }
    return x <= 0.5? pow(2*x, 2*n+1)/2.0 : 1.0+pow(2*(x-1), 2*n+1)/2.0;
}

float QuadraticThroughGivenPoint(float x, float a, float b)
{
    vec2 f = Clamp(a, b);
    float A = (1-f.y)/(1-f.x)-(f.y/f.x);
    float B = (A*(f.x*f.x)-f.y)/f.x;
    float y = A*x*x-B*x;
    y = min(1, max(0, y));
    return y;
}

float Clamp(float a)
{
    float epsilon = 0.00001;
    float min_a = 0.0+epsilon;
    float max_a = 1.0-epsilon;
    return min(max_a, max(min_a, a));
}

float ExponentialEaseIn(float x, float a)
{
    return pow(x, 1/Clamp(a));
}

float ExponentialEaseOut(float x, float a)
{

    return pow(x, Clamp(a));
}

float ExponentialEasing(float x, float a)
{
    a = Clamp(a);
    return a<0.5? pow(x, 2*a) : pow(x, 1/(1-(2*(a-0.5))));
}

float DoubleExponentialSeat(float x, float a)
{
    a = Clamp(a);
    return x <= 0.5? pow(2*x, 1-a)/2.0 : 1.0-pow(2*(1-x), 1-a)/2.0;
}

float DoubleExponentialSigmoid(float x, float a)
{
    a = 1-Clamp(a);
    return x <= 0.5? pow(2*x, 1/a)/2.0 : 1.0-pow(2*(1-x), 1/a)/2.0;
}

float LogisticSigmoid(float x, float a)
{
    a = Clamp(a);
    a = 1.0/(1.0-a)-1.0;
    float A = 1.0/(1+exp(0.0-((x-0.5)*a*2.0)));
    float B = 1.0/(1.0+exp(a));
    float C = 1.0/(1.0+exp(0.0-a));
    return (A-B)/(C-B);
}

float CircularEaseIn(float x)
{
    return 1-sqrt(1-x*x);
}

float CircularEaseOut(float x)
{
    return sqrt(1-(1-x)*(1-x));
}

float DoubleCircelSeat(float x, float a)
{
    a = clamp(a, 0, 1);
    return x <= a? sqrt(a*a-(x-a)*(x-a)) : 1-sqrt((1-a)*(1-a)-(x-a)*(x-a));
}

float DoubleCircleSigmoid(float x, float a)
{
    a = clamp(a, 0, 1);
    return x <= a? a-sqrt(a*a-x*x) : a+sqrt((1-a)*(1-a)-(x-1)*(x-1));
}

float DoubleEllipticSeat(float x, float a, float b)
{
    float epsilon = 0.00001;
    a = clamp(a, epsilon, 1-epsilon);
    b = clamp(b, 0, 1);
    return x <= a? (b/a)*sqrt(a*a-(x-a)*(x-a)): 1-((1-b)/(1-a))*sqrt((1-a)*(1-a)-(x-a)*(x-a));
}

float DoubleEllipticSigmoid(float x, float a, float b)
{
    float epsilon = 0.00001;
    a = clamp(a, epsilon, 1-epsilon);
    b = clamp(b, 0, 1);
    return x <= a? b*(1-sqrt(a*a-x*x)/a) : b+((1-b)/(1-a))*sqrt((1-a)*(1-a)-(x-1)*(x-1));
}

float arcStartAngle;
float arcEndAngle;
float arcStartX, arcStartY;
float arcEndx, arcEndY;
float arcCenterX, arcCenterY;
float arcRadius;

float _LineToPoint(float a, float b, float c, float ptx, float pty)
{
    float d = sqrt(a*a+b*b);
    return d == 0? 0 : (a*ptx+b*pty+c)/d;
}

void _ComputeFilletParameters(float p1x, float p1y, float p2x, float p2y, float p3x, float p3y, float p4x, float p4y, float r)
{
    float c1 = p2x*p1y-p1x*p2y;
    float a1 = p2y-p1y;
    float b1 = p1x-p2x;
    float c2 = p4x*p3y-p3x*p4y;
    float a2 = p4y-p3y;
    float b2 = p3x-p4x;

    if(a1*b2 == a2*b1)
    {
        return;
    }

    float mpx = (p3x+p4x)/2.0;
    float mpy = (p3y+p4y)/2.0;
    float d1 = _LineToPoint(a1, b1, c1, mpx, mpy);
    if(d1 == 0.0)
    {
        return;
    }

    mpx = (p1x+p2x)/2.0;
    mpy = (p1y+p2y)/2.0;
    float d2 = _LineToPoint(a2, b2, c2, mpx, mpy);
    if(d2 == 0.0)
    {
        return;
    }

    float rr = d1 <= 0.0?  -r : r;
    float c1p = c1-rr*sqrt(a1*a1+b1*b1);
    rr = d2 <= 0.0? -r : r;
    float c2p = c2-rr*sqrt(a2*a2+b2*b2);
    float d = (a1*b2)-(a2*b1);

    float pcx = (c2p*b1-c1p*b2)/d;
    float pcy = (c1p*a2-c2p*a1)/d;
    float pax = 0;
    float pay = 0;
    float pbx = 0;
    float pby = 0;
    float dp, cp;
    dp = a1*a1+b1*b1;
    if(dp != 0.0)
    {
        cp = a1*pcy-b1*pcx;
        pax = (-a1*c1-b1*cp)/dp;
        pay = (+a1*cp-b1*c1)/dp;
    }
    dp = a2*a2+b2*b2;
    if(dp != 0.0)
    {
        cp = a2*pcy-b2*pcx;
        pbx = (-a2*c2-b2*cp)/dp;
        pby = (+a2*cp-b2*c2)/dp;
    }

    float gv1x = pax-pcx;
    float gv1y = pay-pcy;
    float gv2x = pbx-pcx;
    float gv2y = pby-pcy;

    float arcStart = atan(gv1y, gv1x);
    float arcAngle = 0.0;
    float dd = sqrt((gv1x*gv1x+gv1y*gv1y)*(gv2x*gv2x+gv2y*gv2y));
    if(dd != 0.0)
    {
        arcAngle = acos((gv1x*gv2x+gv1y*gv2y)/dd);
    }
    float crossproduct = gv1x*gv2y-gv2x*gv1y;
    if(crossproduct<0.0)
    {
        arcStart -= arcAngle;
    }
    float arc1 = arcStart;
    float arc2 = arcStart+arcAngle;
    if(crossproduct<0.0)
    {
        arc1 = arcStart+arcRadius;
        arc2 = arcStart;
    }

    arcCenterX = pcx;
    arcCenterY = pcy;
    arcStartAngle = arc1;
    arcEndAngle = arc2;
    arcRadius = r;
    arcStartX = arcCenterX+arcRadius*cos(arcStartAngle);
    arcStartY = arcCenterY+arcRadius*sin(arcStartAngle);
    arcEndx = arcCenterX+arcRadius*cos(arcEndAngle);
    arcEndY = arcCenterY+arcRadius*sin(arcEndAngle);
}

float DoubleLinearWidthCircularFillet(float x, float a, float b, float r)
{
    float epsilon = 0.00001;
    a = clamp(a, epsilon, 1-epsilon);
    b = clamp(b, epsilon, 1-epsilon);

    _ComputeFilletParameters(0, 0, a, b, a, b, 1, 1, r);
    float t = 0;
    float y = 0;
    x = clamp(x, 0, 1);
    if(x <= arcStartX)
    {
        t = x/arcStartX;
        y = t*arcStartY;
    }
    else if(x >= arcEndx)
    {
        t = (x-arcEndx)/(1-arcEndx);
        y = arcEndY+t*(1-arcEndY);
    }
    else
    {
        if(x >= arcCenterX)
        {
            y = arcCenterY-sqrt(arcRadius*arcRadius-(x-arcCenterX)*(x-arcCenterX));
        }
        else
        {
            y = arcCenterY+sqrt(arcRadius*arcRadius-(x-arcCenterX)*(x-arcCenterX));
        }
    }
    return y;
}

float m_Centerx;
float m_Centery;
float m_dRadius;

bool _Perpendicular(float pt1x, float pt1y, float pt2x, float pt2y, float pt3x, float pt3y)
{
    float xDelta_a = pt2x-pt1x;
    float yDelta_a = pt2y-pt1y;
    float xDelta_b = pt3x-pt2x;
    float yDelta_b = pt3y-pt2y;
    float epsilon = 0.000001;

    if(abs(xDelta_a) <= epsilon && abs(yDelta_b) <= epsilon)
    {
        return false;
    }
    if(abs(yDelta_a) <= epsilon)
    {
        return true;
    }
    else if(abs(yDelta_b) <= epsilon)
    {
        return true;
    }
    else if(abs(xDelta_a) <= epsilon)
    {
        return true;
    }
    else if(abs(xDelta_b) <= epsilon)
    {
        return true;
    }

    return false;
}

void _CalcCircleFrom3Points(float pt1x, float pt1y, float pt2x, float pt2y, float pt3x, float pt3y)
{
    float xDelta_a = pt2x-pt1x;
    float yDelta_a = pt2y-pt1y;
    float xDelta_b = pt3x-pt2x;
    float yDelta_b = pt3y-pt2y;
    float epsilon = 0.000001;

    if(abs(xDelta_a) <= epsilon && abs(yDelta_b) <= epsilon)
    {
        m_Centerx = 0.5*(pt2x+pt3x);
        m_Centery = 0.5*(pt1y+pt2y);
        m_dRadius = sqrt((m_Centerx-pt1x)*(m_Centerx-pt1x)+(m_Centery-pt1y)*(m_Centery-pt1y));
        return;
    }

    float aSlope = yDelta_a/xDelta_a;
    float bSlope = yDelta_b/xDelta_b;
    if(abs(aSlope-bSlope) <= epsilon)
    {
        return;
    }

    m_Centerx = (aSlope*bSlope*(pt1y-pt3y)+bSlope*(pt1x+pt2x)-aSlope*(pt2x+pt3x))/(2*(bSlope-aSlope));
    m_Centery = -1*(m_Centerx-(pt1x+pt2x)/2)/aSlope+(pt1y+pt2y)/2;
    m_dRadius = sqrt((m_Centerx-pt1x)*(m_Centerx-pt1x)+(m_Centery-pt1y)*(m_Centery-pt1y));
}

float CircularArcThroughGivenPoint(float x, float a, float b)
{
    float epsilon = 0.00001;
    a = min(1-epsilon, max(epsilon, a));
    b = min(1-epsilon, max(epsilon, b));
    x = min(1-epsilon, max(epsilon, x));

    float pt1x = 0;
    float pt1y = 0;
    float pt2x = a;
    float pt2y = b;
    float pt3x = 1;
    float pt3y = 1;

    if(!_Perpendicular(pt1x, pt1y, pt2x, pt2y, pt3x, pt3y))
    {
        _CalcCircleFrom3Points(pt1x, pt1y, pt2x, pt2y, pt3x, pt3y);
    }
    else if(!_Perpendicular(pt1x, pt1y, pt3x, pt3y, pt2x, pt2y))
    {
        _CalcCircleFrom3Points(pt1x, pt1y, pt3x, pt3y, pt2x, pt2y);
    }
    else if(!_Perpendicular(pt2x, pt2y, pt1x, pt1y, pt3x, pt3y))
    {
        _CalcCircleFrom3Points(pt2x, pt2y, pt1x, pt1y, pt3x, pt3y);
    }
    else if(!_Perpendicular(pt2x, pt2y, pt3x, pt3y, pt1x, pt1y))
    {
        _CalcCircleFrom3Points(pt2x, pt2y, pt3x, pt3y, pt1x, pt1y);
    }
    else if(!_Perpendicular(pt3x, pt3y, pt1x, pt1y, pt2x, pt2y))
    {
        _CalcCircleFrom3Points(pt3x, pt3y, pt1x, pt1y, pt2x, pt2y);
    }
    else if(!_Perpendicular(pt3x, pt3y, pt2x, pt2y, pt1x, pt1y))
    {
        _CalcCircleFrom3Points(pt3x, pt3y, pt2x, pt2y, pt1x, pt1y);
    }
    else
    {
        return 0;
    }
    

    if(m_Centerx>0&&m_Centerx<1)
    {
        if(a<m_Centerx)
        {
            m_Centerx = 1;
            m_Centery = 0;
            m_dRadius = 1;
        }
        else
        {
            m_Centerx = 0;
            m_Centery = 1;
            m_dRadius = 1;
        }
    }

    float y = 0;
    if(x >= m_Centerx)
    {
        y = m_Centery-sqrt(m_dRadius*m_dRadius-(x-m_Centerx)*(x-m_Centerx));
    }
    else
    {
        y = m_Centery+sqrt(m_dRadius*m_dRadius-(x-m_Centerx)*(x-m_Centerx));
    }
    return y;
}

float QuadraticBezier(float x, float a, float b)
{
    float epsilon = 0.00001;
    a = clamp(a, 0, 1);
    b = clamp(b, 0, 1);
    a += a == 0.5? epsilon : 0;

    float om2a = 1-2*a;
    float om2b = 1-2*b;
    float t = (sqrt(a*a+om2a*x)-a)/om2a;
    return om2b*t*t + 2*b*t;
}

float _SlopeFromT(float t, float a, float b, float c)
{
    return 1.0/(3.0*a*t*t+2.0*b*t+c);
}

float _VFromT(float t, float a, float b, float c, float d)
{
    return a*t*t*t+b*t*t+c*t+d;
}

float CubicBezier(float x, float a, float b, float c, float d)
{
    float x0a = 0.00;
    float y0a = 0.00;
    float x1a = a;
    float y1a = b;
    float x2a = c;
    float y2a = d;
    float x3a = 1.00;
    float y3a = 1.00;

    float A =   x3a-3*x2a+3*x1a-x0a;
    float B = 3*x2a-6*x1a+3*x0a;
    float C = 3*x1a-3*x0a;
    float D =   x0a;

    float E =   y3a-3*y2a+3*y1a-y0a;
    float F = 3*y2a-6*y1a+3*y0a;
    float G = 3*y1a-3*y0a;
    float H =   y0a;

    float currentt = x;
    int nRefinementIterations = 5;
    for(int i=0; i<nRefinementIterations; i++)
    {
        float currentx = _VFromT(currentt, A, B, C, D);
        float currentslope = _SlopeFromT(currentt, A, B, C);
        currentt -= (currentx-x)*(currentslope);
        currentt = clamp(currentt, 0, 1);
    }
    return _VFromT(currentt, E, F, G, H);
}

float _B0(float t)
{
    return (1-t)*(1-t)*(1-t);
}

float _B1(float t)
{
    return 3*t*(1-t)*(1-t);
}

float _B2(float t)
{
    return 3*t*t*(1-t);
}

float _B3(float t)
{
    return t*t*t;
}

float CubicBezierThroughTwoGivenPoints(float x, float a, float b, float c, float d)
{
    float y = 0;
    float epsilon = 0.00001;
    a = clamp(a, epsilon, 1-epsilon);
    b = clamp(b, epsilon, 1-epsilon);

    float x0 = 0;
    float y0 = 0;
    float x1 = 0;
    float y1 = 0;
    float x2 = 0;
    float y2 = 0;
    float x3 = 1;
    float y3 = 1;
    float x4 = a;
    float y4 = b;
    float x5 = c;
    float y5 = d;
    float t1 = 0.3;
    float t2 = 0.7;

    float B0t1 = _B0(t1);
    float B1t1 = _B1(t1);
    float B2t1 = _B2(t1);
    float B3t1 = _B3(t1);
    float B0t2 = _B0(t2);
    float B1t2 = _B1(t2);
    float B2t2 = _B2(t2);
    float B3t2 = _B3(t2);

    float ccx = x4-x0*B0t1-x3*B3t1;
    float ccy = y4-y0*B0t1-y3*B3t1;
    float ffx = x5-x0*B0t2-x3*B3t2;
    float ffy = y5-y0*B0t2-y3*B3t2;

    x2 = (ccx-(ffx*B1t1)/B1t2)/(B2t1-(B1t1*B2t2)/B1t2);
    y2 = (ccy-(ffy*B1t1)/B1t2)/(B2t1-(B1t1*B2t2)/B1t2);
    x1 = (ccx-x2*B2t1)/B1t1;
    y1 = (ccy-y2*B2t1)/B1t1;

    x1 = clamp(x1, epsilon, 1-epsilon);
    x2 = clamp(x2, epsilon, 1-epsilon);

    y = CubicBezier(x, x1, y1, x2, y2);
    return clamp(y, 0, 1);
}

float Impulse(float x, float a)
{
    float h = a*x;
    return h*exp(1.0-h);
}

float CubicPulse(float x, float center, float width)
{
    x = abs(x-center);
    if(x>width)
    {
        return 0.0;
    }
    x /= width;
    return 1.0-x*x*(3.0-2.0*x);
}

float ExponentialStep(float x, float k, float n)
{
    return exp(-k*pow(x, n));
}

float Parabola(float x, float k)
{
    return pow(4.0*x*(1.0-x), k);
}

float PowerCurve(float x, float a, float b)
{
    float k = pow(a+b, a+b)/pow(a, a)/pow(b, b);
    return k*pow(x, a)*pow(1.0-x, b);
}

float LineBrightness(float y0, float y)
{
    return smoothstep(y-0.005, y, y0) - smoothstep(y, y+0.005, y0);
}

void main()
{
    vec4 green = vec4(0.1, 1, 0.2, 1);
    vec2 nc = gl_FragCoord.xy/u_Resolution; // NormalizedCoordinate
    float x = nc.x;
    float y = Linear(nc.x);
    switch(u_ShapingFunction)
    {
        case  1: y = Step(x, 0.5); break;
        case  2: y = SmoothStep(x, 0, 1); break;
        case  3: y = Power(x, 2); break; 
        case  4: y = Sine(x); break;
        case  5: y = Cosine(x); break;
        case  6: y = BlinnWyvillCosineApproximation(x); break;
        case  7: y = DoubleCubicSeat(x, 0.3, 0.5); break;
        case  8: y = DoubleCubicSeatWidthLinearBlend(x, 0.3, 0.5); break;
        case  9: y = DoubleOddPolynomialSeat(x, 0.3, 0.5, 1); break;
        case 10: y = SymmetricDoublePolynomialSigmoids(x, 0.3, 0.5, 1); break;
        case 11: y = QuadraticThroughGivenPoint(x, 0.3, 0.5); break;
        case 12: y = ExponentialEaseIn(x, 0.2); break;
        case 13: y = ExponentialEaseOut(x, 0.2); break;
        case 14: y = ExponentialEasing(x, 0.2); break;
        case 15: y = DoubleExponentialSeat(x, 0.2); break;
        case 16: y = DoubleExponentialSigmoid(x, 0.2); break;
        case 17: y = LogisticSigmoid(x, 0.8); break;
        case 18: y = CircularEaseIn(x); break;
        case 19: y = CircularEaseOut(x); break;
        case 20: y = DoubleCircelSeat(x, 0.5); break;
        case 21: y = DoubleCircleSigmoid(x, 0.5); break;
        case 22: y = DoubleEllipticSeat(x, 0.2, 0.8); break;
        case 23: y = DoubleEllipticSigmoid(x, 0.2, 0.8); break;
        case 24: y = DoubleLinearWidthCircularFillet(x, 0.713, 0.267, 0.25); break;
        case 25: y = CircularArcThroughGivenPoint(x, 0.553, 0.840); break;
        case 26: y = QuadraticBezier(x, 0.94, 0.28); break;
        case 27: y = CubicBezier(x, 0.253, 0.72, 0.75, 0.25); break;
        case 28: y = CubicBezierThroughTwoGivenPoints(x, 0.293, 0.667, 0.75, 0.25); break;
        case 29: y = Impulse(x, 12); break;
        case 30: y = CubicPulse(x, 0.5, 0.1); break;
        case 31: y = ExponentialStep(x, 10.0, 1.0); break;
        case 32: y = Parabola(x, 1.0); break;
        case 33: y = PowerCurve(x, 3.0, 1.0); break;

        default: y = Linear(nc.x);
    }
    vec4 bgColor = vec4(y);
    float brightness = LineBrightness(nc.y, y);
    color = (1-brightness)*bgColor+brightness*green;
}
