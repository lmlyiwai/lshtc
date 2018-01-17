package com.deep;

import static org.junit.Assert.*;

import java.io.IOException;

import org.junit.Test;

public class TestArray {

	@Test
	public void test() throws IOException {
		double[][][] r = new double[3][4][5];
		for(int i = 0; i < r.length; i++) {
			for(int j = 0; j < r[i].length; j++) {
				for(int k = 0; k < r[i][0].length; k++) {
					r[i][j][k] = 1;
				}
			}
		}
		
		double[][][] s = Matrix.scaleMat(r, 0.5);
		for(int i = 0; i < s.length; i++) {
			for(int j = 0; j < s[i].length; j++) {
				for(int k = 0; k < s[i][0].length; k++) {
					System.out.print(s[i][j][k] + " ");
				}
				System.out.println();
			}
			System.out.println();
		}
		System.out.println();
	}

}
