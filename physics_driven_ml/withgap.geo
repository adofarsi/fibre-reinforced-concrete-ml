
// Define the dimensions of the rectangle
Lx = 700;   // length
Ly = 150;   // width
size = 15;
sizeg = 15;
// Create four points of the rectangle
Point(1) = {0, 0, 0, size};  
Point(2) = {Lx, 0, 0, size};
Point(3) = {Lx, Ly, 0, size};
Point(4) = {0, Ly, 0, size};
// left fix point
Point(5) = {45, 0, 0, size};
Point(6) = {55, 0, 0, size};
// right fix point
Point(7) = {645, 0, 0, size};
Point(8) = {655, 0, 0, size};
//top point
Point(9) = {Lx/2-5, Ly, 0, size};
Point(10) = {Lx/2+5, Ly, 0, size};

// Create the four points of the gap
Point(11) = {Lx/2 - 2.5, 0, 0, sizeg};  // lower left corner of the gap
Point(12) = {Lx/2 + 2.5, 0, 0, sizeg};  // lower right corner of the gap
Point(13) = {Lx/2 + 2.5, 50, 0, sizeg};  // Top right corner of the gap
Point(14) = {Lx/2 - 2.5, 50, 0, sizeg};  // top left corner of the gap

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

// Create a line loop, which will define the boundaries of the rectangle
Line Loop(1) = {5,6,13,14,15,16,17,8,9,2, 10,11,12,4};
// Line Loop(18) = {5,6,7,8,9,2, 10,11,12,4};
// Line Loop(19) = {13,14,15,16};

Plane Surface(1) = {1};

// Define the physical line 
Physical Line(1) = {6}; // Set line 5 which is left boundary of the rectangle as tag 1
Physical Line(2) = {8}; // Set line 3 which is right boundary of the rectangle as tag 2
Physical Line(3) = {11}; // Set line 4 which is top boundary of the rectangle as tag 3
// Physical Line(5) = {2};
// Physical Line(6) = {1};
// Physical Line(7) = {3};


Physical Surface(1) = {1};