package com.sparseVector;

/*
 *	���ݵ㣬index-value����ʽ���� 
 */
public class DataPoint {
	public int 		index;
	public double 	value;
	public DataPoint(int index, double value) {
		if(index < 1) {
			System.out.println("index can't below 1");
			return;
		}
		this.index = index;
		this.value = value;
	}
}
