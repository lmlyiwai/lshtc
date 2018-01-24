package com.rssvm;


import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;

import com.tools.Contain;

public class Measures {

	/**
	 * true positive
	 * */
	public static double truePositive(int[] trueLabels, int[] predictLabels) {
		double tp = 0;
		for(int i = 0; i < trueLabels.length; i++) {
			if(trueLabels[i] == 1 && predictLabels[i] == 1) {
				tp++;
			}
		}
		return tp;
	}
	
	/**
	 * false positive ; 预测结果错误，错分类为positive,本为-1类却预测为1类
	 * */
	public static double falsePositive(int[] trueLabels, int[] predictLabels) {
		double fp = 0;
		for(int i = 0; i < trueLabels.length; i++) {
			if(trueLabels[i] == -1 && predictLabels[i] == 1) {
				fp++;
			}
		}
		return fp;
	}
	
	/**
	 * false negative; 预测结果错误，本为+1,取分类为-1
	 * */
	public static double falseNegative(int[] trueLabels, int[] predictLabels) {
		int fn = 0;
		for(int i = 0; i < trueLabels.length; i++) {
			if(trueLabels[i] == 1 && predictLabels[i] == -1) {
				fn++;
			}
		}
		return fn;
	}
	
	/**
	 * true negative
	 * */
	public static double trueNegative(int[] trueLabels, int[] predictLabels) {
		double tn = 0;
		for(int i = 0; i < trueLabels.length; i++) {
			if(trueLabels[i] == -1 && predictLabels[i] == -1) {
				tn++;
			}
		}
		return tn;
	}
	
	public static boolean contain(int[] arr, int value) {
		boolean ifcontain = false;
		if(arr == null) {
			return false;
		}
		for(int i = 0; i < arr.length; i++) {
			if(arr[i] == value) {
				ifcontain = true;
				break;
			}
		} 
		return ifcontain;
	}

	public static double microf1(int[] leaves, int[][] tl, int[][] prel) {
		int[] t = new int[tl.length];
		int[] p = new int[prel.length];
		int id;
		int[] temp1;
		int[] temp2;
		
		int min = t.length > p.length ? p.length : t.length;
		
		double tp = 0;
		double fp = 0;
		double fn = 0;
		double tn = 0;
		
		double tpt = 0;
		double fpt = 0;
		double fnt = 0;
		double tnt = 0;
		
		for(int i = 0; i < leaves.length; i++) {
			id = leaves[i];
			for(int j = 0; j < min; j++) {
				temp1 = tl[j];
				temp2 = prel[j];
				
				if(contain(temp1, id)) {
					t[j] = 1;
				} else {
					t[j] = -1;
				}
				
				if(contain(temp2, id)) {
					p[j] = 1;
				} else {
					p[j] = -1;
				}
			}
			
			if(isAlNegative(t)) {   //样本全为负例，不予考虑
				continue;
			}
			tpt = truePositive(t, p);
			fpt = falsePositive(t, p);
			fnt = falseNegative(t, p);
			tnt = trueNegative(t, p);
			
			tp += tpt;
			fp += fpt;
			fn += fnt;
			tn += tnt;
			
			double pp = 0;
			double rr = 0;
			
			if((tpt + fpt) != 0) {
				pp = tpt / (tpt + fpt);
			}
			
			if((tpt + fnt) != 0) {
				rr = tpt / (tpt + fnt);
			}
			
			double F = 0;
			if((pp + rr) != 0) {
				F = (2 * pp * rr) / (pp + rr);
			}
//System.out.println("label " + id + "  F = " + F + "  tp = " + tpt + "  fp = " + fpt + "  tn = " + tnt + "  fn = " + fnt);
		}	
		
		double precision = tp / (tp + fp);
		double recall = tp / (tp + fn);
		return (2 * precision * recall) / (precision + recall);
	}

	public static double macrof1(int[] leaves, int[][] tl, int[][] prel) {
		int[] t = new int[tl.length];
		int[] p = new int[prel.length];
		int id;
		int[] temp1;
		int[] temp2;
		
		
		double tp = 0;
		double fp = 0;
		double fn = 0;
		
		double precision;
		double recall;
		double mf1 = 0;
		
		double T = leaves.length;
		for(int i = 0; i < leaves.length; i++) {
			id = leaves[i];
			for(int j = 0; j < t.length; j++) {
				temp1 = tl[j];
				temp2 = prel[j];
				
				if(contain(temp1, id)) {
					t[j] = 1;
				} else {
					t[j] = -1;
				}
				
				if(contain(temp2, id)) {
					p[j] = 1;
				} else {
					p[j] = -1;
				}
			}
			
			if(isAlNegative(t)) {          //样本全为负例，不予考虑。
				T--;
				continue;
			}
			
			tp = truePositive(t, p);
			fp = falsePositive(t, p);
			fn = falseNegative(t, p);
			if(tp != 0.0) {
				precision = tp / (tp + fp);
				recall = tp / (tp + fn);
			} else {
				precision = 0.0;
				recall = 0.0;
			}
			
			recall = tp / (tp + fn);
			if(precision != 0.0 && recall != 0.0) {
				mf1 += (2 * precision * recall) / (precision + recall);
			}
		}
		return mf1 / T;
	}
	
	/**
	 *  是否全为负例
	 * */
	public static boolean isAlNegative(int[] rl) {
		boolean flag = true;
		for(int i = 0; i < rl.length; i++) {
			if(rl[i] == 1) {
				flag = false;
				break;
			}
		}
		return flag;
	}
	
	/**
	 * 0/1损失
	 * */
	public static double zeroOneLoss(int[][] trueLabels, int[][] predictLabels) {
		if(trueLabels == null || predictLabels == null || trueLabels.length != predictLabels.length) {
			System.out.println("labels errors.");
			return 1;
		}
		
		double counter = 0;
		for(int i = 0; i < trueLabels.length; i++) {
			if(!sameArray(trueLabels[i], predictLabels[i])) {
				counter++;
			}
		}
		
		return counter / trueLabels.length;
	}
	
	/**
	 * 两数组是否包含相同的值
	 * */
	public static boolean sameArray(int[] a, int[] b) {
		boolean flag = true;
		if(a == null || b == null || a.length != b.length) {
			return false;
		}
		
		int i;
		for(i = 0; i < b.length; i++) {
			if(!Contain.contain(a, b[i])) {
				flag = false;
				break;
			}
		}
		
		return flag;
	}
	
	public static Map<Integer, Integer> labelStatic(int[][] y) {
		Map<Integer, Integer> map = new HashMap<Integer, Integer>();
		
		int[] t = null;
		int label;
		if(y != null) {
			for(int i = 0; i < y.length; i++) {
				t = y[i];
				if(t != null && t.length != 0) {
					for(int j = 0; j < t.length; j++) {
						label = t[j];
						if(map.containsKey(label)) {
							map.put(label, map.get(label) +1);
						} else {
							map.put(label, 1);
						}
					}
				}
			}
		}
		
		return map;
	}

	public static double allNegative(int[][] y) {
		double counter = 0;
		if(y != null) {
			for(int i = 0; i < y.length; i++) {
				if(y[i] == null || y[i].length == 0) {
					counter = counter + 1;
				}
			}
		}
		
		return counter / y.length;
	}
	
	public static int[][] filterLabels(Set<Integer> labelToFilter, int[][] labels) {
		int[][] result = new int[labels.length][];
		List<Integer> list = new ArrayList<Integer>();
		int[] temp = null;
		int i, j;
		for(i = 0; i < labels.length; i++) {
			temp = labels[i];
			list.clear();
			for(j = 0; j < temp.length; j++) {
				if(!labelToFilter.contains(temp[j])) {
					list.add(temp[j]);
				}
			}
			
			
			result[i] = new int[list.size()];
			for(j = 0; j < result[i].length; j++) {
				result[i][j] = list.get(j);
			}
		}
		
		return result;
	}
	
	/**
	 * 
	 */
	public static double symmetricLoss(int[] a, int[] b) {
		if(a == null && b == null) {
			return 0;
		}
		
		if(a == null && b != null) {
			return b.length; 
		}
		
		if(a != null && b == null) {
			return a.length;
		}
		
		
		Map<Integer, Integer> map = new HashMap<Integer, Integer>();
		int key;
		int value;
		
		for(int i = 0; i < a.length; i++) {
			key = a[i];
			if(map.containsKey(key)) {
				value = map.get(key);
				value = value + 1;
				map.put(key, value);
			} else {
				map.put(key, 1);
			}
		}
		
		for(int i = 0; i < b.length; i++) {
			key = b[i];
			if(map.containsKey(key)) {
				value = map.get(key);
				value = value + 1;
				map.put(key, value);
			} else {
				map.put(key, 1);
			}
		}
		
		Set<Integer> set = map.keySet();
		Iterator<Integer> it = set.iterator();
		double counter = 0;
		while(it.hasNext()) {
			key = it.next();
			value = map.get(key);
			if(value == 1) {
				counter++;
			}
		}
		return counter;
	}
	
	/**
	 * 
	 */
	public static double averageSymLoss(int[][] y, int[][] pre) {
		double loss = 0;
		double n = y.length;
		
		int lengtha = y.length;
		int lengthb = pre.length;
		if(lengtha != lengthb) {
			return 0;
		}
		
		double tempLoss = 0;
		for(int i = 0; i < y.length; i++) {
			tempLoss = symmetricLoss(y[i], pre[i]);
			loss = loss + tempLoss;
		}
		return loss / n;
	}
	
}
