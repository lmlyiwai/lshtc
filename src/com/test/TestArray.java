package com.test;

import static org.junit.Assert.*;

import org.junit.Test;

public class TestArray {

	@Test
	public void test() {
		int[][][] a = new int[4][5][6];
		for(int i = 0; i < 4; i++) {
			for(int j = 0; j < 5; j++) {
				for(int k = 0; k < 6; k++) {
					a[i][j][k] = i * j * k;
				}
			}
		}
		
		
		for(int i = 0; i < 4; i++) {
			for(int j = 0; j < 5; j++) {
				for(int k = 0; k < 6; k++) {
					System.out.print(a[i][j][k] + " ");
				}
				System.out.println();
			}
			System.out.println();
		}
	}

}
