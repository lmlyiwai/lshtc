package com.DeepNetworkWithSVM;

import com.sparseVector.DataPoint;

public class SP {
	public static void showMatrix(DataPoint[][] x) {
		if (x == null) {
			return;
		}
		for (int i = 0; i < x.length; i++) {
			for (int j = 0; j < x[i].length; j++) {
				System.out.print(x[i][j].index + ":" + x[i][j].value  + " ");
			}
			System.out.println();
		}
		System.out.println();
	}
	
	public static void showVector(DataPoint[] x) {
		if (x == null) {
			return;
		}
		for (int i = 0; i < x.length; i++) {
			System.out.print(x[i].index + ":" + x[i].value  + " ");
		}
		System.out.println();
	}
}
