package com.tools;

import java.util.Random;

/**
 * �����������
 * */
public class RandomSequence {

	/**
	 * ����0��n֮�䲻�ظ�������У�����0������n
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
	 * 	����һ����������Ĳ��ظ�����
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
