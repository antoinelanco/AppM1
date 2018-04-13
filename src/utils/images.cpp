#include "images.h"
#include <iostream>
#include <sys/types.h>
#include <sys/stat.h>
#include "res.h"

void writeImg(string fileName, vector<float> normalizedRGBArray, int width, int height) {
  char* pathToFeaturesImg = new char[2000]();
  string name = getResFolder() + "/images/";
  sprintf(pathToFeaturesImg, "%s", name.c_str());
  struct stat info;
  if (stat( pathToFeaturesImg, &info ) != 0)
    mkdir(pathToFeaturesImg, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

  FILE *imageFile;
  sprintf(pathToFeaturesImg, "%s", (name + fileName).c_str());
  imageFile = fopen(pathToFeaturesImg, "wb");

  if(imageFile == NULL){
    perror("ERROR: Cannot open output file");
    exit(EXIT_FAILURE);
  }

  fprintf(imageFile, "P6\n");               // P6 filetype
  fprintf(imageFile, "%d %d\n", width, height);   // dimensions
  fprintf(imageFile, "255\n");              // Max pixel

  int nbPixels = width * height * 3;
  unsigned char pix[nbPixels];
  for (int i = 0; i < nbPixels; i++) {
    pix[i] = (unsigned char) 255.f * normalizedRGBArray[i];
  }
  fwrite(pix, 1, nbPixels, imageFile);
  fclose(imageFile);
}
