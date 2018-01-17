package com.test;

import static org.junit.Assert.*;

import org.junit.Test;

import com.tools.Statistic;

public class TestStatistic {

	@Test
	public void test() {
		int[][] y = {{1, 2},
				{1},
				{1, 2},
				{2},
				{1},
				{1, 2}};
		double[][] x = {{1.0, 2.0},
				{3.0, 4.0},
				{5.0, 6.0},
				{7.0, 8.0},
				{9.0, 10.0},
				{11.0, 12.0}};
		
		String[] ul = Statistic.getUniqueLabels(y);
		double[][][] cx = Statistic.getClassVector(x, y, ul);
		for(int i = 0; i < cx.length; i++) {
			System.out.println(ul[i]);
			for(int j = 0; j< cx[i].length; j++) {
				for(int k = 0; k < cx[i][j].length; k++) {
					System.out.print(cx[i][j][k] + " ");
				}
				System.out.println();
			}
			System.out.println();
		}
	}

}
