a = Sqrt(2.); // radius hexagon
t = Sqrt(2.)*0.5; // thickness
angle = Pi/2.0;
phase = Pi/4;
divisions = 20;
nb_pt = 4;
lc = a/divisions;


For i In {1:nb_pt}   // array indexing starts with zero
  Point(i)= {a*Cos(i*angle-phase) ,  a*Sin(i*angle-phase), 0, lc};
  Point(nb_pt+i)= {(a-t)*Cos(i*angle-phase) ,  (a-t)*Sin(i*angle-phase), 0, lc};
EndFor

For i In {1:nb_pt-1}   // array indexing starts with zero
  Line(i)= {i, 1+i};
  Line(nb_pt+i)= {nb_pt+i, nb_pt+1+i};
EndFor

Line(nb_pt)={nb_pt,1};
Line(2*nb_pt)={2*nb_pt,nb_pt+1};

Curve Loop(1) = {2, 3, 4, 1};
//+
Curve Loop(2) = {6, 7, 8, 5};
//+
Plane Surface(1) = {1, 2};

For i In {1:nb_pt} 
  Transfinite Curve(i) = divisions;
EndFor

Physical Curve(1) = {1, 3};
Physical Curve(2) = {2, 4}; 
Physical Surface(1) = {1};
