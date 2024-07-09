// This code was created by pygmsh v6.0.2.
lc0 = 0.1;
lc1 = 0.1;

x0 = -0.5;
y0 = -0.5;
Lx = 1.0;
Ly = 1.0;
r = 0.2;
cx = 0.0;
cy = 0.0;


// Square
p0 = newp;
Point(p0) = {x0, y0, 0.0, lc0};
p1 = newp;
Point(p1) = {x0 + Lx, y0, 0.0, lc0};
p2 = newp;
Point(p2) = {x0 + Lx, y0 + Ly, 0.0, lc0};
p3 = newp;
Point(p3) = {x0, y0 + Ly, 0.0, lc0};
l0 = newl;
Line(l0) = {p0, p1};
l1 = newl;
Line(l1) = {p1, p2};
l2 = newl;
Line(l2) = {p2, p3};
l3 = newl;
Line(l3) = {p3, p0};
ll0 = newll;
Line Loop(ll0) = {l0, l1, l2, l3};


p4 = newp;
Point(p4) = {cx, cy, 0.0, lc1};
p5 = newp;
Point(p5) = {cx + r, cy, 0.0, lc1};
p6 = newp;
Point(p6) = {cx - r, cy, 0.0, lc1};
p7 = newp;
Point(p7) = {cx, cy + r, 0.0, lc1};
p8 = newp;
Point(p8) = {cx, cy - r, 0.0, lc1};


Circle(5) = {6, 5, 8};
//+
Circle(6) = {8, 5, 7};
//+
Circle(7) = {7, 5, 9};
//+
Circle(8) = {9, 5, 6};

Curve Loop(6) = {6, 7, 8, 5};


Plane Surface(3) = {5, 6};
Plane Surface(4) = {6};


Physical Surface(0) = {4};
Physical Surface(1) = {3};
Physical Line(0) = {l0,l1,l2,l3};
