package com.test;

import static org.junit.Assert.*;

import org.junit.Test;

import com.rssvm.Measures;

public class TestSymmetric {

	@Test
	public void test() {
		int[][] a = {{1,2,3, 4}, {2,3,4}, {1,3,4}};
		int[][] b = {{2,3,4, 5}, {1,2,3}, {1,2,3,4}};
		double loss = Measures.averageSymLoss(a, b);
		System.out.println(loss);
	}

}
