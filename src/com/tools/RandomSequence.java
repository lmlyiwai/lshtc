package com.tools;

import java.util.Random;

/**
 * 随机数产生类
 * */
public class RandomSequence {

	/**
	 * 产生0到n之间不重复随机序列，包括0不包括n
	 * */
	public static int[] randomSequence(int n) {
		int[] result = new int[n];
		for(int i = 0; i < n; i++) {
			result[i] = i;
		}
		
		int index;
		int temp;
		for(int i = 0; i < n; i++) {
			index = (int)(Math.random() * n);
			temp = result[i];
			result[i] = result[index];
			result[index] = temp;
		}
		return result;
	}
	
	/**
	 * 	返回一个长度随机的不重复序列
	 * */
	public int[] getMrandomSequence(int n) {
		Random random = new Random();
		int xm = random.nextInt(n);
		
		int[] result = new int[xm];
		int[] rs = randomSequence(n);
		for(int i = 0; i < result.length; i++) {
			result[i] = rs[i];
		}
		return result;
	}
}
