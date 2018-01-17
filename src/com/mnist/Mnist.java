package com.mnist;

import com.fileInputOutput.Parameter;
import com.fileInputOutput.Problem;
import com.rssvm.Linear;
import com.rssvm.Measures;
import com.sparseVector.DataPoint;
import com.sparseVector.SparseVector;
import com.tools.RandomSequence;
import com.tools.Sort;

public class Mnist {
	public static Problem filter(Problem prob, int a, int b) {
		int counter = 0;
		for(int i = 0; i < prob.l; i++) {
			if(prob.y[i][0] == a || prob.y[i][0] == b) {
				counter = counter + 1;
			}
		}
		
		Problem nprob = new Problem();
		nprob.l = counter;
		nprob.n = prob.n;
		nprob.bias = prob.bias;
		nprob.x = new DataPoint[nprob.l][];
		nprob.y = new int[nprob.l][1];
		
		counter = 0;
		for(int i = 0; i < prob.l; i++) {
			if(prob.y[i][0] == a) {
				nprob.x[counter] = prob.x[i];
				nprob.y[counter][0] = 1;
				counter++;
			} else if(prob.y[i][0] == b) {
				nprob.x[counter] = prob.x[i];
				nprob.y[counter][0] = -1;
				counter++;
			}
		}
		return nprob;
	}
	
	public static double[] crossValidation(Problem prob, Parameter param, int n_fold, int k) {
		int n = prob.l;
		
		int[][] pre = new int[n][];
		
		int[] index = RandomSequence.randomSequence(n);
		
		int segLength = n / n_fold;
		
		int vbegin = 0;
		int vend = 0;		
		
		int[] validIndex = null;
		int[] trainIndex = null;
		int counter = 0;
		for(int i = 0; i < n_fold; i++) {
			vbegin = i * segLength;
			vend = i * segLength + segLength;
			
			validIndex = new int[vend - vbegin];
			trainIndex = new int[n - validIndex.length];
			
			counter = 0;
			for(int j = vbegin; j < vend; j++) {
				validIndex[counter++] = index[j];
			}
			
			counter = 0;
			for(int j = 0; j < vbegin; j++) {
				trainIndex[counter++] = index[j];
			}
			for(int j = vend; j < n; j++) {
				trainIndex[counter++] = index[j];
			}
			
			Problem train = new Problem();
			train.l = trainIndex.length;
			train.n = prob.n;
			train.bias = prob.bias;
			train.x = new DataPoint[trainIndex.length][];
			train.y = new int[trainIndex.length][];
			
			counter = 0;
			for(int j = 0; j < trainIndex.length; j++) {
				train.x[counter] = prob.x[trainIndex[j]];
				train.y[counter] = prob.y[trainIndex[j]];
				counter++;
			}
			
			Problem valid = new Problem();
			valid.l = validIndex.length;
			valid.n = prob.n;
			valid.bias = prob.bias;
			valid.x = new DataPoint[validIndex.length][];
			valid.y = new int[validIndex.length][];
			
			counter = 0;
			for(int j = 0; j < validIndex.length; j++) {
				valid.x[counter] = prob.x[validIndex[j]];
				valid.y[counter] = prob.y[validIndex[j]];
				counter++;
			}
			
			double[] loss = new double[1];
			int[] y = new int[train.l];
			for(int m = 0; m < y.length; m++) {
				y[m] = train.y[m][0];
			}
			
//			DataPoint[] w = Linear.train(train, y, param, null, loss, null, 0);
//			
			int[][] predictLabel = new int[valid.l][1];
			int[] validp = knnPredict(train, param, valid.x, k);
			
			for(int m = 0; m < valid.l; m++) {
//				predictLabel[m][0] = SparseVector.innerProduct(w, valid.x[m]) > 0 ? 1:-1;
				predictLabel[m][0] = validp[m];
			}
			
			for(int j = 0; j < validIndex.length; j++) {
				pre[validIndex[j]] = predictLabel[j];
			}
		}
		double hammingloss = Measures.averageSymLoss(prob.y, pre);
		double accuracy = 1 - hammingloss / 2;
		System.out.println("c = " + param.getC() + ", k = " + k + ", accurayc = " + accuracy);
		double[] perf = {hammingloss};
		return perf;
	}
	
	/**
	 * 
	 */
	public static DataPoint[] train(Problem prob, Parameter param) {
		double[] loss = new double[1];
		int[] y = new int[prob.l];
		for(int m = 0; m < y.length; m++) {
			y[m] = prob.y[m][0];
		}
		
		DataPoint[] w = Linear.train(prob, y, param, null, loss, null, 0);
		return w;
	}
	
	public static int[] predict(DataPoint[] w, DataPoint[][] xs) {
		int[] pre = new int[xs.length];
		for(int i = 0; i < pre.length; i++) {
			double inp = SparseVector.innerProduct(xs[i], w);
			if(inp > 0) {
				pre[i] = 1;
			} else {
				pre[i] = -1;
			}
		}
		return pre;
	}
	
	public static double getKnnLabels(Problem prob, Parameter param, int k) {
		DataPoint[] w = train(prob, param);
		double[] pv = new double[prob.l];
		for(int i = 0; i < pv.length; i++) {
			pv[i] = SparseVector.innerProduct(w, prob.x[i]);
		}
		
		int[] pre = new int[prob.l];
		for(int i = 0; i < pre.length; i++) {
			pre[i] = getLabels(pv, pv[i], prob.y, k, 1);
		}
		
		double counter = 0;
		for(int i = 0; i < pre.length; i++) {
			if(pre[i] == prob.y[i][0]) {
				counter++;
			}
		}
		return counter / prob.l;
	}
	
	public static int getLabels(double[] pv, double x, int[][] y, int k, int base) {
		double[] dis = new double[pv.length];
		for(int i = 0; i < dis.length; i++) {
			double sub = pv[i] - x;
			dis[i] = Math.abs(sub);
		}
		
		int[] index = Sort.getIndexBeforeSort(dis);
		double sum = 0;
		for(int i = base; i < base + k; i++) {
			sum += y[index[i]][0];
		}
		if(sum > 0) {
			return 1;
		} else {
			return -1;
		}
	}
	
	public static int[] knnPredict(Problem train, Parameter param, DataPoint[][] x, int k) {
		DataPoint[] w = train(train, param);
		double[] pv = new double[train.l];
		for(int i = 0; i < pv.length; i++) {
			pv[i] = SparseVector.innerProduct(w, train.x[i]);
		}
		
		double[] tpv = new double[x.length];
		for(int i = 0; i < tpv.length; i++) {
			tpv[i] = SparseVector.innerProduct(w, x[i]);
		}
		
		int[] pre = new int[x.length];
		for(int i = 0; i < x.length; i++) {
			pre[i] = getLabels(pv, tpv[i], train.y, k, 0);
		}
		return pre;
	}
}
