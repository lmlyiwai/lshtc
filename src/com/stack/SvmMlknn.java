package com.stack;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;

import com.fileInputOutput.Parameter;
import com.fileInputOutput.Problem;
import com.knn.MLKNN;
import com.rssvm.Linear;
import com.rssvm.Measures;
import com.sparseVector.DataPoint;
import com.sparseVector.SparseVector;
import com.structure.Structure;
import com.tools.Contain;
import com.tools.ProcessProblem;
import com.tools.RandomSequence;
import com.tools.Sort;

public class SvmMlknn {
	private Problem 		prob;
	private Parameter 		param;
	private int[] 			uniqueLabels;
	private DataPoint[][] 	weight;
	private Structure		structure;
	
	public SvmMlknn(Problem prob, Parameter param) {
		this.prob = prob;
		this.param = param;
		this.uniqueLabels = ProcessProblem.getUniqueLabels(prob.y);
	}
	
	public Structure getStructure() {
		return structure;
	}

	public void setStructure(Structure structure) {
		this.structure = structure;
	}

	public DataPoint[][] train(Problem prob, Parameter param) {
		DataPoint[][] w = new DataPoint[this.uniqueLabels.length][];
		int label;
		int[] y;
		double[] tloss = new double[1];
		for(int i = 0; i < this.uniqueLabels.length; i++) {
			label = this.uniqueLabels[i];
			y = getLabels(prob.y, label);
			w[i] = Linear.train(prob, y, param, null, tloss, null, 0);
		}
		return w;
	} 
	

	/**
	 *	获得id指定类标签 
	 */
	public int[] getLabels(int[][] y, int label) {
		if(y == null) {
			return null;
		}
		
		int[] result = new int[y.length];
		for(int i = 0; i < y.length; i++) {
			if(Contain.contain(y[i], label)) {
				result[i] = 1;
			} else {
				result[i] = -1;
			}
		}
		return result;
	}

	/**
	 *	 预测输出值
	 */
	public double[][] predictValues(DataPoint[][] w, DataPoint[][] x) {
		int n = this.uniqueLabels.length;	
		double[][] pv = new double[x.length][n];	
		double[] weight;
		DataPoint[] tx;
		for(int i = 0; i < w.length; i++) {
			weight = SparseVector.sparseVectorToArray(w[i], this.prob.n);
			for(int j = 0; j < x.length; j++) {
				tx = x[j];
				pv[j][i] = SparseVector.innerProduct(weight, tx); 
			}
		}
		return pv;
	}

	/**
	 * 
	 */
	public double[][] predictValues(DataPoint[][] w, double[][] x) {
		int n = this.uniqueLabels.length;
		double[][] pv = new double[x.length][n];
		double[] weight;
		double[] tx;
		for(int i = 0; i < w.length; i++) {
			weight = SparseVector.sparseVectorToArray(w[i], this.prob.n);
			for(int j = 0; j < x.length; j++) {
				tx = x[j];
				pv[j][i] = SparseVector.innerProduct(weight, tx); 
			}
		}
		return pv;
	}
	
	
	/**
	 *	预测类别 
	 */
	public int[][] predict(double[][] pv) {
		int counter = 0;
		int[][] result = new int[pv.length][];
		for(int i = 0; i < pv.length; i++) {
			counter = 0;
			for(int j = 0; j < pv[i].length; j++) {
				if(pv[i][j] > 0) {
					counter++;
				}
			}
			
			result[i] = new int[counter];
			counter = 0;
			for(int j = 0; j < pv[i].length; j++) {
				if(pv[i][j] > 0) {
					result[i][counter++] = this.uniqueLabels[j];
				}
			}
		}
		return result;
	}

	/**
	 * 单类标预测 
	 */
	public int[][] predictMax(double[][] pv) {
		int index = -1;
		double max = Double.NEGATIVE_INFINITY;		
		int[][] result = new int[pv.length][1];
		for(int i = 0; i < pv.length; i++) {
			index = -1;
			max = Double.NEGATIVE_INFINITY;
			for(int j = 0; j < pv[i].length; j++) {
				if(pv[i][j] > max) {
					max = pv[i][j];
					index = j;
				}
			}
			
			result[i] = new int[1];
			result[i][0] = this.uniqueLabels[index];
		}
		return result;
	}
	/**
	 *	全矩阵转换为DataPoint形式 
	 */
	public DataPoint[][] trans(double[][] pv) {
		DataPoint[][] result = new DataPoint[pv.length][];	
		DataPoint temp;
		int index;
		double value;
		int length = pv[0].length;
		int counter = 0;
		for(int i = 0; i < pv.length; i++) {
			
			result[i] = new DataPoint[length];
			counter = 0;
			for(int j = 0; j < pv[i].length; j++) {
				index = j + 1;
				value = pv[i][j];
				temp = new DataPoint(index, value);
				result[i][counter++] = temp;
			}
		}
		return result;
	}
	
	
	public int[] getUniqueLabels() {
		return uniqueLabels;
	}

	public void setUniqueLabels(int[] uniqueLabels) {
		this.uniqueLabels = uniqueLabels;
	}
	
	/**
	 *	加入一列 
	 */
	public void extendWithBias(double[][] pv, double bias) {
		double[] temp = null;
		int length = 0;
		int counter = 0;
		for(int i = 0; i < pv.length; i++) {
			length = pv[i].length + 1;
			temp = new double[length];
			counter = 0;
			for(int j = 0; j < pv[i].length; j++) {
				temp[counter++] = pv[i][j];
			}
			temp[counter] = bias;
			pv[i] = temp;
		}
	}

	/**
	 *	向量归一化 
	 */
	public void scale(double[][] pv) {
		double norm = 0;
		double[] temp = null;
		for(int i = 0; i < pv.length; i++) {
			temp = pv[i];
			norm = SparseVector.innerProduct(temp, temp);
			norm = Math.pow(norm, 0.5);
			for(int j = 0; j < pv[i].length; j++) {
				pv[i][j] = pv[i][j] / norm;
			}
		}
	}
	
	/**
	 *	交叉验证 ，返回预测类标
	 */
	public int[][] crossValidation(Problem prob, Parameter param, int n_fold) {
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
			
			DataPoint[][] w = train(train, param);
			
			double[][] pv = predictValues(w, valid.x);
			int[][] predictLabel = predict(pv);
			
			for(int j = 0; j < validIndex.length; j++) {
				pre[validIndex[j]] = predictLabel[j];
			}
		}
		double microf1 = Measures.microf1(this.uniqueLabels, prob.y, pre);
		double macrof1 = Measures.macrof1(this.uniqueLabels, prob.y, pre);
		double hammingloss = Measures.averageSymLoss(prob.y, pre);
		System.out.println("c = " + param.getC() + ", Micro-F1 = " + microf1 + ", Macro-F1 = " 
				+ macrof1 + ", Hamming Loss = " + hammingloss);
		return pre;
	}
	
	/**
	 * c惩罚, k近邻, n_fold交叉验证
	 */
	public double[][] gridSerach(Problem prob, Parameter param, int n_fold, double c, int[] k) {
		int n = prob.l;
		
		int[][][] pre = new int[k.length][n][];
		
		int[] index = RandomSequence.randomSequence(n);
		
		int segLength = n / n_fold;
		
		int vbegin = 0;
		int vend = 0;		
		
		param.setC(c);
		
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
			
			DataPoint[][] w = train(train, param);
			
			double[][] trainpv = predictValues(w, train.x);
			scale(trainpv);
			train.x = trans(trainpv);
			
			
			double[][] validpv = predictValues(w, valid.x);
			scale(validpv);
			valid.x = trans(validpv);

			MLKNN knn = new MLKNN(train);
			int[][][] temppre = new int[k.length][valid.l][];
			for(int h = 0; h < k.length; h++) {
				knn.getStatistic(train, k[h], 1);
				temppre[h] = knn.predict(train, valid.x);
			}
			
			for(int h = 0; h < valid.l; h++) {
				for(int m = 0; m < k.length; m++) {
					pre[m][validIndex[h]] = temppre[m][h];
				}
			}
		}
		
		double[][] performance = new double[k.length][2];
		for(int i = 0; i < k.length; i++) {
			double microf1 = Measures.microf1(this.uniqueLabels, prob.y, pre[i]);
			double macrof1 = Measures.macrof1(this.uniqueLabels, prob.y, pre[i]);
			performance[i][0] = microf1;
			performance[i][1] = macrof1;		
			double hammingLoss = Measures.averageSymLoss(prob.y, pre[i]);
			double zeroneloss = Measures.zeroOneLoss(prob.y, pre[i]);
			System.out.println("C = " + c + ", K = " + k[i] + 
					", Micro-F1 = " + microf1 + ", Macro-F1 = " + macrof1 + 
					", Hamming Loss = " + hammingLoss + 
					", zero one loss = " + zeroneloss);
		}
		return performance;
	}
	
	/**
	 * 
	 */
	public int[][] predict(DataPoint[][] xs, int k) {
		double[][] trainpv = predictValues(this.weight, this.prob.x);
		scale(trainpv);
		this.prob.x = trans(trainpv);
		
		
		double[][] validpv = predictValues(this.weight, xs);
		scale(validpv);
		DataPoint[][] test = trans(validpv);
		
		MLKNN knn = new MLKNN(this.prob);
		knn.getStatistic(this.prob, k, 1);
		int[][] y = knn.predict(this.prob, test);
		return y;
	}

	public DataPoint[][] getWeight() {
		return weight;
	}

	public void setWeight(DataPoint[][] weight) {
		this.weight = weight;
	}
	
}
