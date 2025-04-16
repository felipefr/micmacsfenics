a = 1.0;
eta = 8;
b = a/Sqrt(3.); // radius hexagon
t_article = a/eta;
t = t_article/Sqrt(3.); // thickness
angle = Pi/3.0;
phase = -Pi/6.0;
divisions = 100;
lc = a/divisions;

For i In {1:6}   // array indexing starts with zero
  Point(i)= {b*Cos(i*angle + phase) ,  b*Sin(i*angle + phase), 0, lc};
  Point(6+i)= {(b-t)*Cos(i*angle + phase) ,  (b-t)*Sin(i*angle + phase), 0, lc};
EndFor

For i In {1:5}   // array indexing starts with zero
  Line(i)= {i, 1+i};
  Line(6+i)= {6+i, 7+i};
EndFor

Line(6)={6,1};
Line(12)={12,7};

Curve Loop(1) = {2, 3, 4, 5, 6, 1};
//+
Curve Loop(2) = {8, 9, 10, 11, 12, 7};
//+
Plane Surface(1) = {1, 2};

short_sides = {2, 4}; 
 
For i In {1:6} 
  Transfinite Curve(i) = divisions;
EndFor

Physical Curve(1) = {1, 4};
Physical Curve(2) = {2, 5};
Physical Curve(3) = {3, 6}; 
Physical Surface(1) = {1};
