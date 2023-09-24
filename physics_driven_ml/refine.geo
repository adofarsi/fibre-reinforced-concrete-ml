
// Define the dimensions of the rectangle
Lx = 700;   // length
Ly = 150;   // width
size = 10;
sizeg = 5;
// Create four points of the rectangle
Point(1) = {0, 0, 0, size};  
Point(2) = {Lx, 0, 0, size};
Point(3) = {Lx, Ly, 0, size};
Point(4) = {0, Ly, 0, size};
// left fix point
Point(5) = {49, 0, 0, size};
Point(6) = {51, 0, 0, size};
// right fix point
Point(7) = {649, 0, 0, size};
Point(8) = {651, 0, 0, size};
//top point
Point(9) = {Lx/2-1, Ly, 0, size};
Point(10) = {Lx/2+1, Ly, 0, size};

// Create the four points of the gap
Point(11) = {Lx/2 - 2.5, 0, 0, 10};  // lower left corner of the gap
Point(12) = {Lx/2 + 2.5, 0, 0, 10};  // lower right corner of the gap
Point(13) = {Lx/2 + 2.5, 50, 0, 10};  // Top right corner of the gap
Point(14) = {Lx/2 - 2.5, 50, 0, 10};  // top left corner of the gap

// Refining the grid along the centerline
Point(15) = {Lx/2 - 0.01, 50.01, 0, sizeg-1};
Point(16) = {Lx/2 - 0.01, Ly - 0.01, 0, sizeg-1}; 
Point(17) = {Lx/2 + 0.01, Ly - 0.01, 0, sizeg-1};
Point(18) = {Lx/2 + 0.01, 50.01, 0, sizeg-1};

// Line of the rectangle
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

Line(5) = {1,5};
Line(6) = {5,6};
Line(7) = {6,7};
Line(8) = {7,8};
Line(9) = {8,2};
Line(10) = {3,10};
Line(11) = {10,9};
Line(12) = {9,4};
Line(13) = {6, 11};
Line(14) = {11, 14};
Line(15) = {14, 13};
Line(16) = {13, 12};

Line(17) = {12, 7};

Line(18) = {15, 16};
Line(19) = {16, 17};
Line(20) = {17, 18};
Line(21) = {18, 15};

// Create a line loop, which will define the boundaries of the rectangle
Line Loop(1) = {5,6,13,14,15,16,17,8,9,2, 10,11,12,4};
Line Loop(2) = {18,19,20,21};


Plane Surface(1) = {1,2};
Plane Surface(2) = {2};

// Define the physical line 
Physical Line(1) = {6}; // Set line 5 which is left boundary of the rectangle as tag 1
Physical Line(2) = {8}; // Set line 3 which is right boundary of the rectangle as tag 2
Physical Line(3) = {11}; // Set line 4 which is top boundary of the rectangle as tag 3
// Physical Line(5) = {2};
// Physical Line(6) = {1};
// Physical Line(7) = {3};


Physical Surface(1) = {1};
Physical Surface(2) = {2};