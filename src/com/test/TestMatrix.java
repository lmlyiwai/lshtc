package com.test;

import static org.junit.Assert.*;

import org.junit.Test;

import com.tools.Matrix;

public class TestMatrix {

	@Test
	public void test() {
		double[][] a = {{1.5532, 1.2622, 1.2720, 1.4888},
						{1.2622, 1.2293, 1.1170, 1.2506},
						{1.2720, 1.1170, 1.2278, 1.3250},
						{1.4888, 1.2506, 1.3250, 1.8226}};
//		double[][] a = {{1,22,34,22},
//						{1,11,5,21},
//						{0,1,5,11},
//						{7,2,13,19}};
		double deta = Matrix.det(a);
		System.out.println(deta);
		
		double[][] inva = Matrix.inv(a);
		double[][] mul = Matrix.multi(a, inva);
		for(int i = 0; i < mul.length; i++) {
			for(int j = 0; j < mul[i].length; j++) {
				System.out.print(mul[i][j] + " ");
			}
			System.out.println();
		}
		System.out.println();
		
		for(int i = 0; i < inva.length; i++) {
			for(int j = 0; j < inva[i].length; j++) {
				System.out.print(inva[i][j] + " ");
			}
			System.out.println();
		}
		System.out.println();
	}

}
