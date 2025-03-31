#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define NUM_ASSETS 20
#define NUM_ITERATIONS 1000
#define MAX_LINE_LENGTH 1024

// Function to compare two doubles for qsort
int compare(const void *a, const void *b) {
  double da = *(const double *)a;
  double db = *(const double *)b;
  return (da > db) - (da < db);
}

// Function to read predicted_returns from CSV file
void read_predicted_returns(double *predicted_returns, const char *filename) {
  FILE *file = fopen(filename, "r");
  if (file == NULL) {
    perror("Error opening file");
    exit(1);
  }

  for (int i = 0; i < NUM_ASSETS; i++) {
    if (fscanf(file, "%lf", &predicted_returns[i]) != 1) {
      fprintf(stderr, "Error reading predicted_returns at index %d\n", i);
      exit(1);
    }
    // printf("predicted_returns[%d] = %f\n", i, predicted_returns[i]); // Debug
    // statement
  }

  fclose(file);
}

// Function to read cov_matrix from CSV file
void read_cov_matrix(double cov_matrix[NUM_ASSETS][NUM_ASSETS],
                     const char *filename) {
  FILE *file = fopen(filename, "r");
  if (file == NULL) {
    perror("Error opening file");
    exit(1);
  }

  char line[MAX_LINE_LENGTH];
  int row = 0;

  while (fgets(line, sizeof(line), file)) {
    // Print the entire line for debugging
    // printf("Read line: %s", line);

    char *token = strtok(line, ",");
    for (int col = 0; col < NUM_ASSETS; col++) {
      if (token == NULL) {
        fprintf(stderr, "Error: Not enough values in row %d\n", row);
        exit(1);
      }
      cov_matrix[row][col] = atof(token);
      // printf("cov_matrix[%d][%d] = %f\n", row, col, cov_matrix[row][col]); //
      // Debug statement
      token = strtok(NULL, ",");
    }
    if (token != NULL) {
      fprintf(stderr, "Error: Extra data in row %d\n", row);
      exit(1);
    }
    row++;
    if (row >= NUM_ASSETS) {
      break;
    }
  }

  if (row < NUM_ASSETS) {
    fprintf(stderr, "Error: Not enough rows in cov_matrix\n");
    exit(1);
  }

  fclose(file);
}

int main() {
  srand(time(NULL)); // Seed the random number generator

  double predicted_returns[NUM_ASSETS];
  double cov_matrix[NUM_ASSETS][NUM_ASSETS];

  // Read the data from CSV files
  read_predicted_returns(predicted_returns, "predicted_returns.csv");
  read_cov_matrix(cov_matrix, "cov_matrix.csv");

  double *random_profit = (double *)malloc(NUM_ITERATIONS * sizeof(double));
  double *random_risk = (double *)malloc(NUM_ITERATIONS * sizeof(double));
  int *random_nonzero = (int *)malloc(NUM_ITERATIONS * sizeof(int));

  if (random_profit == NULL || random_risk == NULL || random_nonzero == NULL) {
    perror("Error allocating memory");
    exit(1);
  }

  for (int k = 0; k < NUM_ITERATIONS; k++) {
    double points[21];
    points[0] = 0.0;
    points[20] = 1.0;
    for (int i = 1; i < 20; i++) {
      points[i] = (double)rand() / RAND_MAX;
    }

    qsort(points, 21, sizeof(double), compare);

    double weights[NUM_ASSETS];
    for (int i = 0; i < NUM_ASSETS; i++) {
      weights[i] = points[i + 1] - points[i];
    }

    double profit = 0.0;
    double risk = 0.0;
    int nonzero = 0;
    for (int i = 0; i < NUM_ASSETS; i++) {
      profit += weights[i] * predicted_returns[i];
      if (weights[i] >= 0.01) {
        nonzero += 1;
      }
      for (int j = 0; j < NUM_ASSETS; j++) {
        risk += weights[i] * weights[j] * cov_matrix[i][j];
      }
    }
    random_profit[k] = profit;
    random_risk[k] = risk;
    random_nonzero[k] = nonzero;
  }

  // Write the results to a file
  FILE *file = fopen("output.csv", "w");
  if (file == NULL) {
    perror("Error opening file");
    free(random_profit);
    free(random_risk);
    free(random_nonzero);
    return 1;
  }

  for (int k = 0; k < NUM_ITERATIONS; k++) {
    fprintf(file, "%f,%f,%d\n", random_profit[k], random_risk[k],
            random_nonzero[k]);
  }

  fclose(file);
  free(random_profit);
  free(random_risk);

  return 0;
}
