package com.matrix;

import org.junit.Test;

import com.tools.Matrix;


public class TestMatrix {

	@Test
	public void test() {
		double[][] matrixA = {{2.0, 10.0, 3.0, 5.0},
				{10.0, 11.0, 5.0, 4.0},
				{2.0, 3.0, 9.0, 3.0},
				{5.0, 12.0, 6.0, 7.0}};
		double[][] matrixB = {{-1.0, 4.0, 3.0, 10.0},
				{2.0, -4.0, 8.0, 10.0},
				{1.0, 5.0, 6.0, 7.0},
				{10.0, 9.0, 4.0, 5.0}};
		double deta = Matrix.det(matrixA);
		double detb = Matrix.det(matrixB);
		System.out.println("det a = " + deta + ", det b = " + detb);
		
		double[][] mulAB = Matrix.multi(matrixA,  matrixB);
		Matrix.showMatrix(mulAB);
		double[][] inva = Matrix.inv(matrixA);
		Matrix.showMatrix(inva);
		double[][] invb = Matrix.inv(matrixB);
		Matrix.showMatrix(invb);
	}

}
