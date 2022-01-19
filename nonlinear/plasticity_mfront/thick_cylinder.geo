Ri = 1.;
Re = 1.3;
d = 0.03;

Point(0) = {0, 0, 0, d};
Point(1) = {Ri, 0, 0, d};
Point(2) = {Re, 0, 0, d};
Point(3) = {0, Re, 0, d};
Point(4) = {0, Ri, 0, d};
Line(1) = {1, 2};

Circle(2) = {2, 0, 3};
Line(3) = {3, 4};
Circle(4) = {4, 0, 1};
Line Loop(5) = {4, 1, 2, 3};
Plane Surface(6) = {5};

Physical Line(1) = {1};
Physical Line(2) = {2};
Physical Line(3) = {3};
Physical Line(4) = {4};
Physical Surface(1) = {6};
