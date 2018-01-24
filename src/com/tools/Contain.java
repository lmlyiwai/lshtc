package com.tools;

public class Contain {
	/**
	 * 	�ж�array���Ƿ����value
	 * */
	public static boolean contain(int[] array, int value) {
		boolean flag = false;
		if(array == null) {
			return flag;
		}
		
		for(int i = 0; i < array.length; i++) {
			if(array[i] == value) {
				flag = true;
				break;
			}
		}
		return flag;
	} 
	
	/**
	 * �ж�arraya�Ƿ����arrayb
	 * */
	public static boolean contain(int[] arraya, int[] arrayb) {
		boolean flag = true;
		if(arraya == null || arrayb == null) {
			flag = false;
			return false;
		}
		
		for(int i = 0; i < arrayb.length; i++) {
			if(!contain(arraya, arrayb[i])) {
				flag = false;
				break;
			}
		}
		return flag;
	}
	
	/**
	 * a �Ƿ����b�е�ĳЩԪ��
	 * */
	public static boolean subcontain(int[] a, int[] b) {
		if(a == null || b == null) {
			return false;
		}
		
		boolean flag = false;
		for(int i = 0; i < b.length; i++) {
			if(contain(a, b[i])) {
				flag = true;
				break;
			}
		}
		return flag;
	}
}
