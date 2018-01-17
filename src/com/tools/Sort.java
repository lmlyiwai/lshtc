package com.tools;

public class Sort {
	
	/**
	 *	包括start和end在内 
	 */
	public static void quickSort(double[] numbers, int[] index, int start, int end) {   
        if (start < end) {   
            double base = numbers[start]; // 选定的基准值（第一个数值作为基准值）   
            double temp; // 记录临时中间值   
            int ind;
            int i = start, j = end;   
            do {   
                while ((numbers[i] < base) && (i < end))   
                    i++;   
                while ((numbers[j] > base) && (j > start))   
                    j--;   
                if (i <= j) {   
                    temp = numbers[i];   
                    numbers[i] = numbers[j];   
                    numbers[j] = temp;   
                    
                    ind = index[i];
                    index[i] = index[j];
                    index[j] = ind;
                    
                    i++;   
                    j--;   
                }   
            } while (i <= j);   
            if (start < j)   
                quickSort(numbers, index, start, j);   
            if (end > i)   
                quickSort(numbers, index, i, end);   
        }   
    }   
	
	
	public static void quickSort(double[] numbers, int start, int end) {   
        if (start < end) {   
            double base = numbers[start]; // 选定的基准值（第一个数值作为基准值）   
            double temp; // 记录临时中间值   
            int i = start, j = end;   
            do {   
                while ((numbers[i] < base) && (i < end))   
                    i++;   
                while ((numbers[j] > base) && (j > start))   
                    j--;   
                if (i <= j) {   
                    temp = numbers[i];   
                    numbers[i] = numbers[j];   
                    numbers[j] = temp;                      
                    i++;   
                    j--;   
                }   
            } while (i <= j);   
            if (start < j)   
                quickSort(numbers, start, j);   
            if (end > i)   
                quickSort(numbers, i, end);   
        }   
    }   
	
	/**
	 * 排序后各值在排序之前数组中的位置
	 * */
	public static int[] getIndexBeforeSort(double[] a) {
		if(a == null) {
			return null;
		}
		
		double[] ta = new double[a.length];
		int[] result = new int[ta.length];
		int i;
		for(i = 0; i < ta.length; i++) {
			ta[i] = a[i];
			result[i] = i;
		}
		
		quickSort(ta, result, 0, ta.length - 1);
		
		return result;
	}

	public static int[] readperm(int n) {
		int[] result = new int[n];
		int i, j;
		for(i = 0; i < result.length; i++) {
			result[i] = i;
		}
		
		int temp;
		for(i = 0; i < result.length; i++) {
			j = (int)(Math.random() * result.length);
			temp = result[i];
			result[i] = result[j];
			result[j] = temp;
		}
		return result;
	}
	
	public static double[] sort(double[] nums) {
		double[] result = new double[nums.length];
		for(int i = 0; i < nums.length; i++) {
			result[i] = nums[i];
		}
		
		quickSort(result, 0, result.length - 1);
		return result;
	}
	
	public static int[] sort(int[] nums) {
		double[] result = new double[nums.length];
		for(int i = 0; i < result.length; i++) {
			result[i] = nums[i];
		}
		
		quickSort(result, 0, result.length - 1);
		
		int[] re = new int[result.length];
		for(int i = 0; i < re.length; i++) {
			re[i] = (int)result[i];
		}
		return re;
	}
}
