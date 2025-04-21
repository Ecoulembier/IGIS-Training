/*
This script is used as a macro in ImageJ (Plugins --> Macros --> Run) to obtain the coordinates of
boxes drawn around all plants. These coordinates are saved to a file called "crop_coorinates_plate#.txt"
with # being the number of the plate it refers to (asked at start of macro). This should be done for all
10 plates so 10 files with coordinates are created, and preferabley be saved inside 1 folder.
These coordinates are thus obtained from the reference plate (oldest picture of plate X) since the
younger plants will always be inside of this box, which is crucial for the next cropping step.
IMPORTANT: Do not forget to crop the reference pictures as well, since these need to be used here!
IMPORTANT: When drawing the boxes, hold shift to create square boxes in ImageJ!
*/

macro "Get 12 Rectangle Coordinates" {
    // Ask for the index number (0-9)
    indexNumber = getNumber("Enter the index number (0-9):", 0);
    if (indexNumber < 0 || indexNumber > 9) {
        exit("Index number must be between 0 and 9");
    }
    
    // Create an array to store coordinates
    coordinates = newArray(12*6);
    
    // Loop 12 times to get rectangles
    for (i = 0; i < 12; i++) {
        // Ask user to draw rectangle
        run("Select None");
        setTool("rectangle");
        waitForUser("Draw Rectangle", "Draw rectangle #" + (i+1) + " and click OK");
        
        // Check if a selection exists
        if (selectionType() == -1) {
            showMessage("Error", "No selection was made. Please try again.");
            i--; // Decrement i to retry this iteration
            continue;
        }
        
        // Get rectangle coordinates
        getSelectionBounds(x, y, width, height);
        
        // Store coordinates in array
        coordinates[i*6] = indexNumber;
        coordinates[i*6 + 1] = i + 1;
        coordinates[i*6 + 2] = x;
        coordinates[i*6 + 3] = y;
        coordinates[i*6 + 4] = width;
        coordinates[i*6 + 5] = height;
    }
    
    // Write coordinates to file
    outputPath = getDirectory("Choose a Directory");
    fileName = "crop_coordinates_plate" + indexNumber + ".txt";
    f = File.open(outputPath + fileName);
    for (i = 0; i < 12; i++) {
        line = d2s(coordinates[i*6], 0) + "," + d2s(coordinates[i*6 + 1], 0) + "," + 
               d2s(coordinates[i*6 + 2], 0) + "," + d2s(coordinates[i*6 + 3], 0) + "," + 
               d2s(coordinates[i*6 + 4], 0) + "," + d2s(coordinates[i*6 + 5], 0);
        print(f, line);
    }
    File.close(f);
    
    showMessage("Coordinates saved", "The coordinates have been saved to:\n" + outputPath + fileName);
}
