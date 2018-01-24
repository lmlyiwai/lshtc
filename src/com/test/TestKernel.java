package com.test;

import static org.junit.Assert.*;

import org.junit.Test;

import com.tools.Kernel;

public class TestKernel {

	@Test
	public void test() {
		double[] a = {0.3, 0.4, 0.5, 0.6, 0.8};
		double[] t = Kernel.polynomial(a);
		double sum = 0;
		for(int i = 0; i < t.length; i++) {
			System.out.print(t[i] + " ");
			sum += t[i] * t[i];
		}
		System.out.println();
		System.out.print("sum = " + sum);
	}

}
