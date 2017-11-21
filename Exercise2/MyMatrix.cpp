//Krzysztof Borawski 238152 Inf rok III grupa I
#include <eigen3/Eigen/Dense>

#include <iostream>
#include <type_traits>
#include <vector>
#include <cmath>
#include <chrono>

using namespace Eigen;

struct HelpValues {
	HelpValues() {
		size = 10;
		maxPrecision = 17;
	}
	int size;
	int maxPrecision;
};

template<typename T>
class MyMatrix {

public:

	MyMatrix(int row_Count, int column_Count) {
		rowCount = row_Count;
		columnCount = column_Count;
		matrixVector.resize(rowCount);
		for (auto &r : matrixVector)
			r.resize(columnCount);
	};

	MyMatrix() { };

	MyMatrix<T>& operator+(const MyMatrix<T>& other) {

		if (rowCount != other.rowCount || columnCount != other.columnCount) {
			std::cout << "Not equal value of rowCount and columnCount between matrix'es returning w/out changes" << std::endl;
			return *this;
		}

		for (int i = 0; i < rowCount; i++) {
			for (int j = 0; j < columnCount; j++)
				matrixVector[i][j] = matrixVector[i][j] + other.matrixVector[i][j];
		}

		return *this;
	}

	MyMatrix<T>& operator*(const MyMatrix<T>& other) {

		if (columnCount != other.rowCount) {
			std::cout << "Not equal value of A ColumnCount != B rowCount returning" << std::endl;
			return *this;
		}

		//Dzia�a ju� w 100%
		MyMatrix<T> matrixC(rowCount, columnCount);
		matrixC.matrixVector = matrixVector;
		matrixVector.resize(rowCount);

		for (auto &r : matrixVector)
			r.resize(other.columnCount);

		columnCount = other.columnCount;
		rowCount = matrixVector.size();

		for (int i = 0; i < matrixC.rowCount; i++) {
			for (int j = 0; j < other.columnCount; j++) {
				matrixVector[i][j] = 0;
				for (int k = 0; k < matrixC.columnCount; k++) {
					matrixVector[i][j] += matrixC.matrixVector[i][k] * other.matrixVector[k][j];
				}
			}
		}

		return *this;
	}

	void printMatrix() {
		//method to print Matrix of doubles and floats

		for (int i = 0; i < rowCount; i++) {
			for (int j = 0; j < columnCount; j++) {
				printf(" %20.17E; ", matrixVector[i][j]);
			}
			std::cout << std::endl;
		}

	}

	int rowCount;
	int columnCount;
	std::vector<std::vector<T>> matrixVector;

	void passMatrixToMyMatrix(MatrixXd matrix) {

		for (int i = 0; i < rowCount; i++) {
			for (int j = 0; j < columnCount; j++) {
				matrixVector[i][j] = static_cast<T>(matrix(i, j));
			}
		}

	}

	void passVectorToMyMatrix(VectorXd vector) {

		for (int i = 0; i < rowCount; i++) {
			matrixVector[i][0] = static_cast<T>(vector(i));
		}

	}
};

void MatrixTimesVectorTest(MatrixXd &eigenMatrix, VectorXd &eigenVector, int size) {
	//Calculating value of operation Matrix * Vector and checking error vs eigen
	MyMatrix<double> myOwnMatrixDoubl(size, size);
	MyMatrix<double> myOwnVectorDoubl(size, 1);
	MyMatrix<float> myOwnMatrixFloat(size, size);
	MyMatrix<float> myOwnVectorFloat(size, 1);

	myOwnMatrixDoubl.passMatrixToMyMatrix(eigenMatrix);
	myOwnMatrixFloat.passMatrixToMyMatrix(eigenMatrix);
	myOwnVectorDoubl.passVectorToMyMatrix(eigenVector);
	myOwnVectorFloat.passVectorToMyMatrix(eigenVector);

	auto eigen_start_time = std::chrono::high_resolution_clock::now();
	MatrixXd eigenResult = eigenMatrix*eigenVector;
	auto eigen_end_time = std::chrono::high_resolution_clock::now();
	auto eigen_time = std::chrono::duration_cast<std::chrono::microseconds>(eigen_end_time - eigen_start_time).count();

	std::cout << "EigenTime: " << eigen_time << std::endl;

	auto start_time = std::chrono::high_resolution_clock::now();
	auto resDouble = myOwnMatrixDoubl * myOwnVectorDoubl;
	//auto resFloat = myOwnMatrixFloat * myOwnVectorFloat;
	auto end_time = std::chrono::high_resolution_clock::now();
	auto my_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();

	std::cout << "MyTime: " << my_time << std::endl;
	std::cout << "Error: " << std::endl;

	for (int i = 0; i < myOwnMatrixDoubl.rowCount; i++) {
		for (int j = 0; j < myOwnMatrixDoubl.columnCount; j++) {
			printf(" %20.17E ", abs(myOwnMatrixDoubl.matrixVector[i][j] - eigenResult(i, j)));
		}
		std::cout << std::endl;
	}

}

void AddMatrixTimesVectorTest(MatrixXd &eigenMatA, MatrixXd &eigenMatB, MatrixXd &eigenMatC, VectorXd &eigenVector, int size) {
	//Calculating value of operation (MatrixA + MatrixB + MatrixC) * Vector
	MyMatrix<double> myOwnMatrixA(size, size), myOwnMatrixB(size, size), myOwnMatrixC(size, size), myOwnVector(size, 1);
	MyMatrix<float> matrixA_Float(size, size), matrixB_Float(size, size), matrixC_Float(size, size), vectorFloat(size, 1);

	myOwnVector.passVectorToMyMatrix(eigenVector);
	vectorFloat.passVectorToMyMatrix(eigenVector);
	myOwnMatrixA.passMatrixToMyMatrix(eigenMatA);
	matrixA_Float.passMatrixToMyMatrix(eigenMatA);
	myOwnMatrixB.passMatrixToMyMatrix(eigenMatB);
	matrixB_Float.passMatrixToMyMatrix(eigenMatB);
	myOwnMatrixC.passMatrixToMyMatrix(eigenMatC);
	matrixC_Float.passMatrixToMyMatrix(eigenMatC);
	

	auto eigen_start_time = std::chrono::high_resolution_clock::now();
	MatrixXd addTimesResult = (eigenMatA + eigenMatB + eigenMatC) * eigenVector;
	auto eigen_end_time = std::chrono::high_resolution_clock::now();
	auto eigen_time = std::chrono::duration_cast<std::chrono::microseconds>(eigen_end_time - eigen_start_time).count();

	std::cout << "EigenTime: " << eigen_time << std::endl;

	auto start_time = std::chrono::high_resolution_clock::now();
	auto res = (myOwnMatrixA + myOwnMatrixB + myOwnMatrixC) * myOwnVector;
	//auto resFloat = (matrixA_Float + matrixB_Float + matrixC_Float) * vectorFloat;
	auto end_time = std::chrono::high_resolution_clock::now();
	auto my_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();

	std::cout << "MyTime: " << my_time << std::endl;
	std::cout << "Error: " << std::endl;

	for (int i = 0; i < myOwnMatrixA.rowCount; i++) {
		for (int j = 0; j < myOwnMatrixA.columnCount; j++) {
			myOwnMatrixA.matrixVector[i][j] = abs(myOwnMatrixA.matrixVector[i][j] - addTimesResult(i, j));
			//matrixA_Float.matrixVector[i][j] = abs(matrixA_Float.matrixVector[i][j] - (float)addTimesResult(i, j));
		}
	}

	myOwnMatrixA.printMatrix();
	//matrixA_Float.printMatrix();
}

void MultiplyMatrix(MatrixXd &eigenMatrixA, MatrixXd &eigenMatrixB, MatrixXd &eigenMatrixC, int size) {
	//Calculating value of MatrixA * ( MatrixB * MatrixC)
	MyMatrix<double> myOwnMatrixA(size, size), myOwnMatrixB(size, size), myOwnMatrixC(size, size);
	MyMatrix<float> matrixA_Float(size, size), matrixB_Float(size, size), matrixC_Float(size, size);

	myOwnMatrixA.passMatrixToMyMatrix(eigenMatrixA);
	matrixA_Float.passMatrixToMyMatrix(eigenMatrixA);
	myOwnMatrixB.passMatrixToMyMatrix(eigenMatrixB);
	matrixB_Float.passMatrixToMyMatrix(eigenMatrixB);
	myOwnMatrixC.passMatrixToMyMatrix(eigenMatrixC);
	matrixC_Float.passMatrixToMyMatrix(eigenMatrixC);

	auto eigen_start_time = std::chrono::high_resolution_clock::now();
	MatrixXd eigenResult = eigenMatrixA * (eigenMatrixB * eigenMatrixC);
	auto eigen_end_time = std::chrono::high_resolution_clock::now();
	auto eigen_time = std::chrono::duration_cast<std::chrono::microseconds>(eigen_end_time - eigen_start_time).count();

	std::cout << "EigenTime: " << eigen_time << std::endl;

	auto start_time = std::chrono::high_resolution_clock::now();
	auto res = myOwnMatrixA * (myOwnMatrixB * myOwnMatrixC);
	//auto resfloat = matrixA_Float * (matrixB_Float * matrixC_Float);
	auto end_time = std::chrono::high_resolution_clock::now();
	auto my_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();

	std::cout << "MyTime: " << my_time << std::endl;
	std::cout << "MatrixA * (MatrixB * MatrixC Error: " << std::endl;

	for (int i = 0; i < myOwnMatrixA.rowCount; i++) {
		for (int j = 0; j < myOwnMatrixA.columnCount; j++) {
			myOwnMatrixA.matrixVector[i][j] = abs(myOwnMatrixA.matrixVector[i][j] - eigenResult(i, j));
			//matrixA_Float.matrixVector[i][j] = abs(matrixA_Float.matrixVector[i][j] - (float)eigenResult(i, j));
		}
	}

	myOwnMatrixA.printMatrix();
	//matrixA_Float.printMatrix();

}

template <typename T>
MyMatrix<T> gaussPartialPiv(MyMatrix<T> matrix) {
	int n = matrix.rowCount;

	for (int i = 0; i < n; i++) {

		//Searching for maximum element in this column
		T maxValue = abs(matrix.matrixVector[i][i]);
		int maxRow = i;

		for (int k = i + 1; k < n; k++) {
			if (abs(matrix.matrixVector[k][i]) > maxValue) {
				maxValue = abs(matrix.matrixVector[k][i]);
				maxRow = k;
			}
		}

		//Swaping maximum row with current row
		std::vector<T> &v1 = matrix.matrixVector[maxRow];
		std::vector<T> &v2 = matrix.matrixVector[i];
		v1.swap(v2);


		//Make all rows below this one 0 in current column
		for (int k = i + 1; k < n; k++) {
			T result = -matrix.matrixVector[k][i] / matrix.matrixVector[i][i];
			for (int j = i; j < n + 1; j++) {
				if (i == j) {
					matrix.matrixVector[k][j] = 0;
				}
				else {
					matrix.matrixVector[k][j] += result * matrix.matrixVector[i][j];
				}
			}
		}

	}
	// Solving equaation Ax=B for an upper triangular matrix A
	MyMatrix<T> ResultVector(n, 1);

	for (int i = n - 1; i >= 0; i--) {
		ResultVector.matrixVector[i][0] = matrix.matrixVector[i][n] / matrix.matrixVector[i][i];
		for (int k = i - 1; k >= 0; k--) {
			matrix.matrixVector[k][n] -= matrix.matrixVector[k][i] * ResultVector.matrixVector[i][0];
		}
	}

	return ResultVector;
}

template <typename T>
MyMatrix<T> gauss(MyMatrix<T> matrix) {
	int n = matrix.rowCount;

	for (int i = 0; i < n; i++) {
		//Make all rows below this one 0 in current column
		for (int k = i + 1; k < n; k++) {
			T result = -matrix.matrixVector[k][i] / matrix.matrixVector[i][i];
			for (int j = i; j < n + 1; j++) {
				if (i == j) {
					matrix.matrixVector[k][j] = 0;
				}
				else {
					matrix.matrixVector[k][j] += result * matrix.matrixVector[i][j];
				}
			}
		}
	}
	// Solving equaation Ax=B for an upper triangular matrix A
	MyMatrix<T> ResultVector(n, 1);
	for (int i = n - 1; i >= 0; i--) {
		ResultVector.matrixVector[i][0] = matrix.matrixVector[i][n] / matrix.matrixVector[i][i];
		for (int k = i - 1; k >= 0; k--) {
			matrix.matrixVector[k][n] -= matrix.matrixVector[k][i] * ResultVector.matrixVector[i][0];
		}
	}
	return ResultVector;
}

template <typename T>
MyMatrix<T> gaussFullPiv(MyMatrix<T> matrix) {
	int n = matrix.rowCount;

	std::vector<int> order;
	order.resize(n);
	for (int i = 0; i < n; i++) {
		order[i] = i;
	}

	for (int i = 0; i < n; i++) {

		T maxValue = abs(matrix.matrixVector[i][i]);
		int rowIndex = i;
		int columnIndex = i;

		//Searching for greatest element in whole matrix
		for (int k = i; k < n; k++) {
			for (int j = i; j < n; j++) {
				if (abs(matrix.matrixVector[k][j]) > maxValue) {
					maxValue = abs(matrix.matrixVector[k][j]);
					rowIndex = k;
					columnIndex = j;
				}
			}
		}

		//Swaping rows
		if (i != rowIndex) {
			for (int k = 0; k <= n; k++) {
				T tmp = matrix.matrixVector[i][k];
				matrix.matrixVector[i][k] = matrix.matrixVector[rowIndex][k];
				matrix.matrixVector[rowIndex][k] = tmp;
			}
		}

		//Swaping columns
		if (i != columnIndex) {
			int tmp = order[i];
			order[i] = order[columnIndex];
			order[columnIndex] = tmp;

			for (int k = 0; k < n; k++) {
				T temp = matrix.matrixVector[k][i];
				matrix.matrixVector[k][i] = matrix.matrixVector[k][columnIndex];
				matrix.matrixVector[k][columnIndex] = temp;
			}

		}

		for (int k = i + 1; k < n; k++) {
			T result = -matrix.matrixVector[k][i] / matrix.matrixVector[i][i];
			for (int j = i; j < n + 1; j++) {
				if (i == j) {
					matrix.matrixVector[k][j] = 0;
				}
				else {
					matrix.matrixVector[k][j] += result * matrix.matrixVector[i][j];
				}
			}
		}

	}

	// Solving equaation Ax=B for an upper triangular matrix A
	MyMatrix<T> ResultVector(n, 1);

	for (int i = n - 1; i >= 0; i--) {
		ResultVector.matrixVector[i][0] = matrix.matrixVector[i][n] / matrix.matrixVector[i][i];
		for (int k = i - 1; k >= 0; k--) {
			matrix.matrixVector[k][n] -= matrix.matrixVector[k][i] * ResultVector.matrixVector[i][0];
		}
	}

	//Puting ResultVector in order
	MyMatrix<T> temporaryVector(n, 1);

	for (int i = 0; i < ResultVector.rowCount; i++) {
		for (int j = 0; j < ResultVector.columnCount; j++) {
			temporaryVector.matrixVector[i][j] = ResultVector.matrixVector[i][j];
		}
	}
	for (int i = 0; i < ResultVector.rowCount; i++) {
		ResultVector.matrixVector[order[i]][0] = temporaryVector.matrixVector[i][0];
	}

	return ResultVector;
}

void gaussTests(MatrixXd &eigenMatrix, VectorXd &eigenVector, int size) {
	MyMatrix<double> MyADouble(size, size + 1);
	MyMatrix<float> MyAFloat(size, size + 1);
	//Generating a copy of a matrix A and adding X vector as a last column

	for (int i = 0; i < MyADouble.rowCount; i++) {
		MyADouble.matrixVector[i][MyADouble.rowCount] = eigenVector(i);
		MyAFloat.matrixVector[i][MyAFloat.rowCount] = (float)eigenVector(i);
		for (int j = 0; j < MyADouble.rowCount; j++) {
			MyADouble.matrixVector[i][j] = eigenMatrix(i, j);
			MyAFloat.matrixVector[i][j] = (float)eigenMatrix(i, j);
		}
	}

	auto eigen_partial_start_time = std::chrono::high_resolution_clock::now();
	MatrixXd eigenPartialResultVector = eigenMatrix.partialPivLu().solve(eigenVector);
	auto eigen_partial_end_time = std::chrono::high_resolution_clock::now();
	auto eigen_partial_time = std::chrono::duration_cast<std::chrono::microseconds>(eigen_partial_end_time - eigen_partial_start_time).count();

	auto eigen_full_start_time = std::chrono::high_resolution_clock::now();
	MatrixXd eigenFullResultVector = eigenMatrix.fullPivLu().solve(eigenVector);
	auto eigen_full_end_time = std::chrono::high_resolution_clock::now();
	auto eigen_full_time = std::chrono::duration_cast<std::chrono::microseconds>(eigen_full_end_time - eigen_full_start_time).count();

	std::cout << "Full;" << eigen_full_time << ";Partial;" << eigen_partial_time << std::endl;

	//Calculating time of Double operations
	auto double_start_time = std::chrono::high_resolution_clock::now();
	MyMatrix<double> doubleResVector = gaussPartialPiv(MyADouble);
	//MyMatrix<double> doubleResVector = gauss(MyADouble);
	//MyMatrix<double> doubleResVector = gaussFullPiv(MyADouble);
	auto double_end_time = std::chrono::high_resolution_clock::now();
	auto double_time = std::chrono::duration_cast<std::chrono::microseconds>(double_end_time - double_start_time).count();

	//Calculating time of float operations
	auto float_start_time = std::chrono::high_resolution_clock::now();
	MyMatrix<float> floatResVector = gaussPartialPiv(MyAFloat);
	//MyMatrix<float> floatResVector = gauss(MyAFloat);
	//MyMatrix<float> floatResVector = gaussFullPiv(MyAFloat);
	auto float_end_time = std::chrono::high_resolution_clock::now();
	auto float_time = std::chrono::duration_cast<std::chrono::microseconds>(float_end_time - float_start_time).count();

	std::cout << "Double;" << double_time << ";Float;" << float_time << std::endl;

	printf("ErrDouble\n");

	for (int i = 0; i < doubleResVector.rowCount; i++) {
		for (int j = 0; j < doubleResVector.columnCount; j++) {
			printf(" %20.17E", abs(doubleResVector.matrixVector[i][j] - eigenPartialResultVector(i, j)));
		}
		std::cout << std::endl;
	}

}

int main() {

	HelpValues h;
	std::cout.precision(h.maxPrecision);
	MatrixXd eigenMatrixA = MatrixXd::Random(h.size, h.size);
	MatrixXd eigenMatrixB = MatrixXd::Random(h.size, h.size);
	MatrixXd eigenMatrixC = MatrixXd::Random(h.size, h.size);
	VectorXd eigenVector = VectorXd::Random(h.size);
	//MatrixTimesVectorTest(eigenMatrixA, eigenVector, h.size);
	//AddMatrixTimesVectorTest(eigenMatrixA, eigenMatrixB, eigenMatrixC, eigenVector, h.size);
	//MultiplyMatrix(eigenMatrixA, eigenMatrixB, eigenMatrixC, h.size);
	gaussTests(eigenMatrixA, eigenVector, h.size);
	getchar();
	return 0;
}
