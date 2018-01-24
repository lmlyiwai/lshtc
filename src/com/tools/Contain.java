package com.tools;

public class Contain {
	/**
	 * 	判断array中是否包含value
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
	 * 判断arraya是否包含arrayb
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
	 * a 是否包含b中的某些元素
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
