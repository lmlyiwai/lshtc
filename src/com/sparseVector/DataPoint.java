package com.sparseVector;

/*
 *	数据点，index-value的形式给出 
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
