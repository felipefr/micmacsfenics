//+
L = 1.0;
Lcube = 0.4*L;
x0 = 0.0;
y0 = 0.0;
z0 = 0.0;
R = 0.25;
lc = 0.5*L;
lc2 = 0.5*L;

Point(1) = {-L/2, -L/2, -L/2, lc};
Point(2) = {L/2, -L/2, -L/2, lc};
Point(3) = {L/2, L/2, -L/2, lc};
Point(4) = {-L/2, L/2, -L/2, lc};

Point(5) = {-L/2, -L/2, L/2, lc};
Point(6) = {L/2, -L/2, L/2, lc};
Point(7) = {L/2, L/2, L/2, lc};
Point(8) = {-L/2, L/2, L/2, lc};

Point(9) = {x0 - Lcube/2, y0 - Lcube/2, z0 - Lcube/2, lc2};
Point(10) = {x0 + Lcube/2, y0 - Lcube/2, z0 - Lcube/2, lc2};
Point(11) = {x0 + Lcube/2, y0 + Lcube/2, z0 - Lcube/2, lc2};
Point(12) = {x0 - Lcube/2, y0 + Lcube/2, z0 - Lcube/2, lc2};

Point(13) = {x0 - Lcube/2, y0 - Lcube/2, z0 + Lcube/2, lc2};
Point(14) = {x0 + Lcube/2, y0 - Lcube/2, z0 + Lcube/2, lc2};
Point(15) = {x0 + Lcube/2, y0 + Lcube/2, z0 + Lcube/2, lc2};
Point(16) = {x0 - Lcube/2, y0 + Lcube/2, z0 + Lcube/2, lc2};

//+
//Physical Surface(1) = {1,2,3,4};
//+
//Physical Volume(1) = {1};
// Gmsh project created on Thu Feb 13 18:49:11 2025
//+
Line(1) = {2, 3};
//+
Line(2) = {3, 4};
//+
Line(3) = {4, 8};
//+
Line(4) = {8, 5};
//+
Line(5) = {5, 6};
//+
Line(6) = {6, 2};
//+
Line(7) = {2, 1};
//+
Line(8) = {1, 5};
//+
Line(9) = {4, 1};
//+
Line(10) = {3, 7};
//+
Line(11) = {7, 8};
//+
Line(12) = {7, 6};

//+
Line(13) = {10, 11};
//+
Line(14) = {11, 12};
//+
Line(15) = {15, 11};
//+
Line(16) = {12, 16};
//+
Line(17) = {16, 15};
//+
Line(18) = {15, 14};
//+
Line(19) = {14, 13};
//+
Line(20) = {13, 16};
//+
Line(21) = {9, 13};
//+
Line(22) = {9, 10};
//+
Line(23) = {10, 14};
//+
Line(24) = {9, 12};
//+
Curve Loop(1) = {4, 5, -12, 11};
//+
Plane Surface(1) = {1};
//+
Curve Loop(2) = {6, 7, 8, 5};
//+
Plane Surface(2) = {2};
//+
Curve Loop(3) = {10, 11, -3, -2};
//+
Plane Surface(3) = {3};
//+
Curve Loop(4) = {1, 2, 9, -7};
//+
Plane Surface(4) = {4};
//+
Curve Loop(5) = {6, 1, 10, 12};
//+
Plane Surface(5) = {5};
//+
Curve Loop(6) = {4, -8, -9, 3};
//+
Plane Surface(6) = {6};
//+
Curve Loop(7) = {23, -18, 15, -13};
//+
Plane Surface(7) = {7};
//+
Curve Loop(8) = {14, 16, 17, 15};
//+
Plane Surface(8) = {8};
//+
Curve Loop(9) = {20, 17, 18, 19};
//+
Curve Loop(10) = {24, 16, -20, -21};
//+
Plane Surface(9) = {10};
//+
Curve Loop(11) = {19, -21, 22, 23};
//+
Plane Surface(10) = {11};
//+
Plane Surface(11) = {9};
//+
Curve Loop(12) = {14, -24, 22, 13};
//+
Plane Surface(12) = {12};
//+
Surface Loop(1) = {11, 9, 12, 8, 7, 10};
//+
Volume(1) = {1};
//+
Surface Loop(2) = {5, 2, 4, 3, 1, 6};
//+
Volume(2) = {1, 2};
//+
Physical Volume(0) = {1}; // inclusion
//+
Physical Volume(1) = {2}; // matrix
//+
Physical Surface(1) = {6, 1, 3, 4, 5, 2};
