#include "CloudTracker.h"
#include <iostream>

#ifdef _WIN32
#include <windows.h>
#include <commdlg.h>

std::string openFileDialog(const std::string& title) {
    char filename[MAX_PATH];
    OPENFILENAMEA ofn;
    ZeroMemory(&filename, sizeof(filename));
    ZeroMemory(&ofn, sizeof(ofn));
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = NULL;
    ofn.lpstrFilter = "Image Files\0*.png;*.jpg;*.jpeg;*.bmp;*.tif\0Any File\0*.*\0";
    ofn.lpstrFile = filename;
    ofn.nMaxFile = MAX_PATH;
    ofn.lpstrTitle = title.c_str();
    ofn.Flags = OFN_DONTADDTORECENT | OFN_FILEMUSTEXIST;
  
    if (GetOpenFileNameA(&ofn)) {
        return std::string(filename);
    }
    return "";
}
#else
// Fallback for non-Windows (if they compile on another OS)
std::string openFileDialog(const std::string& title) {
    std::string filename;
    std::cout << title << ": ";
    std::getline(std::cin, filename);
    return filename;
}
#endif

int main() {
    std::cout << "=== Satellite Image Cloud Tracker ===" << std::endl;
    
    std::string path1, path2;

#ifdef _WIN32
    std::cout << "Please select the first satellite image (T1)..." << std::endl;
    path1 = openFileDialog("Select First Image (Time T1)");
    if (path1.empty()) {
        std::cerr << "No file selected. Exiting." << std::endl;
        return 1;
    }
    
    std::cout << "Please select the second satellite image (T2)..." << std::endl;
    path2 = openFileDialog("Select Second Image (Time T2)");
    if (path2.empty()) {
        std::cerr << "No file selected. Exiting." << std::endl;
        return 1;
    }
#else
    std::cout << "Enter path to first image: ";
    std::cin >> path1;
    std::cout << "Enter path to second image: ";
    std::cin >> path2;
#endif

    std::cout << "\nImages selected:" << std::endl;
    std::cout << "T1: " << path1 << std::endl;
    std::cout << "T2: " << path2 << std::endl;

    double timeDifference = 1.0;
    std::cout << "\nEnter time difference between images (e.g., 1.5 hours): ";
    std::cin >> timeDifference;

    CloudTracker tracker;
    if (tracker.loadImages(path1, path2)) {
        std::cout << "Processing images... Please wait." << std::endl;
        tracker.process(timeDifference);
        
        std::cout << "Processing complete. Displaying results." << std::endl;
        tracker.showResults();
    } else {
        std::cout << "Failed to process images." << std::endl;
    }

    return 0;
}
