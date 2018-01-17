package com.flatSvm;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import javax.sound.sampled.ReverbType;
import javax.xml.crypto.Data;

import com.fileInputOutput.Parameter;
import com.fileInputOutput.Problem;
import com.rssvm.Linear;
import com.rssvm.Measures;
import com.sparseVector.DataPoint;
import com.sparseVector.SparseVector;
import com.structure.Structure;
import com.tools.Contain;
import com.tools.Matrix;
import com.tools.ProcessProblem;
import com.tools.RandomSequence;
import com.tools.Sort;

public class FlatSVM {
	private Problem 		prob;
	private Parameter 		param;
	private int[] 			ulabels;
	private DataPoint[][] 	w;
	private Structure       tree;
	private Map<Integer, int[]> pathes;
	private double 			maxTreeDis;
	private double[][]		classCenter;
	private double[][]      means;
	private double[][]      stds;
	
	public FlatSVM(Problem prob, Parameter param) {
		this.prob = prob;
		this.param = param;
		this.ulabels = ProcessProblem.getUniqueLabels(prob.y);
	}
	
	public DataPoint[][] train(Problem prob, Parameter param) {
		DataPoint[][] weight = new DataPoint[this.ulabels.length][];
		for(int i = 0; i < this.ulabels.length; i++) {
			int label = this.ulabels[i];
			int[] labels = getBinaryLabels(prob.y, label);
			double[] tloss = new double[1];
			System.out.print(i + ", " + label + ", ");
			long start = System.currentTimeMillis();
			weight[i] = Linear.train(prob, labels, param, null, tloss, null, 0);
			long end = System.currentTimeMillis();
			System.out.println((end - start) + "ms");
		}
		this.w = weight;
		return weight;
	}
	
	
	/**
	 *  每训练一个权将其写入文件
	 * @throws IOException 
	 */
	public void train(Problem prob, Parameter param, String wfile) throws IOException {
		DataPoint[] weight = null;
		BufferedWriter out = new BufferedWriter(
				new OutputStreamWriter(new FileOutputStream(wfile)));
		String line = null;
		for(int i = 0; i < this.ulabels.length; i++) {
			int label = this.ulabels[i];
			int[] labels = getBinaryLabels(prob.y, label);
			double[] tloss = new double[1];
			System.out.print(i + ", " + label + ", ");
			long start = System.currentTimeMillis();
			weight = Linear.train(prob, labels, param, null, tloss, null, 0);
		
			line = new String();
			for(int j = 0; j < weight.length; j++) {
				line = line + weight[j].index + ":"  + weight[j].value + " ";
			}
			line = line + "\n";
			out.write(line);
			long end = System.currentTimeMillis();
			System.out.println((end - start) + "ms");
		}
		out.close();
	}
	
	/**
	 *  每训练一个权将其写入文件
	 * @throws IOException 
	 */
	public void trainWithDelta(Problem prob, Parameter param, String wfile) throws IOException {
		DataPoint[] weight = null;
		BufferedWriter out = new BufferedWriter(
				new OutputStreamWriter(new FileOutputStream(wfile)));
		String line = null;
		for(int i = 0; i < this.ulabels.length; i++) {
			int label = this.ulabels[i];
			double[] cost = getDelta(label, prob.y);
			int[] labels = getBinaryLabels(prob.y, label);
			double[] tloss = new double[1];
			System.out.print(i + ", " + label + ", ");
			long start = System.currentTimeMillis();
			weight = Linear.train(prob, labels, param, null, cost, tloss, null, 0);
			line = new String();
			for(int j = 0; j < weight.length; j++) {
				line = line + weight[j].index + ":"  + weight[j].value + " ";
			}
			line = line + "\n";
			out.write(line);
			long end = System.currentTimeMillis();
			System.out.println((end - start) + "ms");
		}
		out.close();
	}
	
	public double[] getDelta(int label, int[][] labels) {
		int[] path = this.tree.getPathToRoot(label);
		double[] cost = new double[labels.length];
		int[] tpath = null;
		for(int i = 0; i < labels.length; i++) {
			tpath = this.tree.getPathToRoot(labels[i][0]);
			cost[i] = getDelta(path, tpath);
		}
		return cost;
	}
	
	public double getDelta(int[] path, int[] pb) {
		if(path[0] == pb[0]) {
			return (double)path.length;
		} else {
			int p1 = path.length - 1;
			int p2 = pb.length - 1;
			while(p1 >= 0 && p2 >= 0) {
				if(path[p1] == pb[p2]) {
					p1--;
					p2--;
				} else {
					break;
				}
			}
			p1 += 1;
			return (double)p1;
		}
	}
	
	/**
	 *  每训练一个权将其写入文件
	 * @throws IOException 
	 */
	public void train(Problem prob, Parameter param, String wfile, double cp, double cn) throws IOException {
		DataPoint[] weight = null;
		
		BufferedWriter out = new BufferedWriter(
				new OutputStreamWriter(new FileOutputStream(wfile)));
		String line = null;
		for(int i = 0; i < this.ulabels.length; i++) {
			int label = this.ulabels[i];
			int[] labels = getBinaryLabels(prob.y, label);
			double[] tloss = new double[1];
			System.out.print(i + ", " + label + ", ");
			long start = System.currentTimeMillis();
			double[] cost = getMargin(labels, cp, cn);
			weight = Linear.train(prob, labels, param, null, cost, tloss, null, 0);
			line = new String();
			for(int j = 0; j < weight.length; j++) {
				line = line + weight[j].index + ":"  + weight[j].value + " ";
			}
			line = line + "\n";
			out.write(line);
			long end = System.currentTimeMillis();
			System.out.println((end - start) + "ms");
		}
		out.close();
	}
	
	public double[] getMargin(int[] labels, double cp, double cn) {
		double[] cost = new double[labels.length];
		for(int i = 0; i < cost.length; i++) {
			if(labels[i] == 1) {
				cost[i] = cp;
			} else {
				cost[i] = cn;
			}
		}
		return cost;
	}
	
	/**
	 *  每训练一个权将其写入文件
	 * @throws IOException 
	 */
	public void trainWithInnerNode(Problem prob, Parameter param, String wfile) throws IOException {
		int[] nodes = this.tree.levelTraverse();
		
		DataPoint[] weight = null;
		
		BufferedWriter out = new BufferedWriter(
				new OutputStreamWriter(new FileOutputStream(wfile)));
		String line = null;
		for(int i = 0; i < nodes.length; i++) {
			int label = nodes[i];
			int[] labels = constructLabels(prob.y, label);
			double[] tloss = new double[1];
			System.out.print(i + ", " + label + ", ");
			long start = System.currentTimeMillis();
			weight = Linear.train(prob, labels, param, null, tloss, null, 0);
		
			line = new String();
			for(int j = 0; j < weight.length; j++) {
				line = line + weight[j].index + ":"  + weight[j].value + " ";
			}
			line = line + "\n";
			out.write(line);
			long end = System.currentTimeMillis();
			System.out.println((end - start) + "ms");
		}
		out.close();
	}
	
	/**
	 * 为中间设置类标
	 */
	public int[] constructLabels(int[][] y, int label) {
		int[] des = this.tree.getDescendent(label);
		int[] labels = new int[y.length];
		for(int i = 0; i < y.length; i++) {
			if(Contain.contain(des, y[i][0])) {
				labels[i] = 1;
			} else {
				labels[i] = -1;
			}
		}
		return labels;
	}
	
	/**
	 *  每训练一个权将其写入文件
	 * @throws IOException 
	 */
	public void train(double[][] pv, int[][] y, Parameter param, String wfile) throws IOException {
		DataPoint[] weight = null;
		
		BufferedWriter out = new BufferedWriter(
				new OutputStreamWriter(new FileOutputStream(wfile)));
		String line = null;
		for(int i = 0; i < this.ulabels.length; i++) {
			int label = this.ulabels[i];
			int[] labels = getBinaryLabels(prob.y, label);
			double[] tloss = new double[1];
			System.out.print(i + ", " + label + ", ");
			long start = System.currentTimeMillis();
			weight = Linear.train(pv, labels, param, null, tloss, null, 0);
			line = new String();
			for(int j = 0; j < weight.length; j++) {
				line = line + weight[j].index + ":"  + weight[j].value + " ";
			}
			line = line + "\n";
			out.write(line);
			long end = System.currentTimeMillis();
			System.out.println((end - start) + "ms");
		}
		out.close();
	}
	
	/**
	 *  每训练一个权将其写入文件
	 * @throws IOException 
	 */
	public void trainWithCost(Problem prob, Parameter param, String wfile) throws IOException {
		DataPoint[] weight = null;
		
		BufferedWriter out = new BufferedWriter(
				new OutputStreamWriter(new FileOutputStream(wfile)));
		String line = null;
		for(int i = 0; i < this.ulabels.length; i++) {
			int label = this.ulabels[i];
			System.out.print(i + ", " + label + ", ");
			long start = System.currentTimeMillis();
			int[] labels = getBinaryLabels(prob.y, label);
//			double[] cost = getCost(prob.y, label, this.maxTreeDis);
			double[] margin = getMargin(prob.y, label, this.maxTreeDis);

			double[] tloss = new double[1];
//			weight = Linear.train(prob, labels, param, null, cost, tloss, null, 0);
			weight = Linear.train(prob, labels, param, margin);
			line = new String();
			start = System.currentTimeMillis();
			for(int j = 0; j < weight.length; j++) {
				line = line + weight[j].index + ":"  + weight[j].value + " ";
			}
			line = line + "\n";
			out.write(line);
			long end = System.currentTimeMillis();
			System.out.println((end - start) + "ms");
		}
		out.close();
	}
	
	/**
	 * 
	 */
	public double[] getCost(int[][] y, int label, double maxDis) {
		double[] cost = new double[y.length];
		for(int i = 0; i < y.length; i++) {
			if(Contain.contain(y[i], label)) {
				cost[i] = maxDis;
			} else {
				int[] patha = this.pathes.get(label);
				int[] pathb = this.pathes.get(y[i][0]);
				double dis = this.tree.getDistance(patha, pathb);
				cost[i] = Math.max((maxDis - dis), 1);
			}
		}
		return cost;
	}
	
	/**
	 * 
	 */
	public int[] getBinaryLabels(int[][] y, int label) {
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
	 * 
	 */
	public double[] getMargin(int[][] y, int label, double maxDis) {
		double[] cost = new double[y.length];
		for(int i = 0; i < y.length; i++) {
			if(Contain.contain(y[i], label)) {
				cost[i] = maxDis / 2;
			} else {
				cost[i] = 1;
			}
		}
		return cost;
	}
	
	public double[] crossValidation(Problem prob, Parameter param, int n_fold) {
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
			
			train(train, param);
			
			int[][] predictLabel = predict(valid.x);
			
			for(int j = 0; j < validIndex.length; j++) {
				pre[validIndex[j]] = predictLabel[j];
			}
		}
		double microf1 = Measures.microf1(this.ulabels, prob.y, pre);
		double macrof1 = Measures.macrof1(this.ulabels, prob.y, pre);
		double hammingloss = Measures.averageSymLoss(prob.y, pre);
		System.out.println("c = " + param.getC() + ", Micro-F1 = " + microf1 + ", Macro-F1 = " 
				+ macrof1 + ", Hamming Loss = " + hammingloss);
		double[] perf = {microf1, macrof1, hammingloss};
		return perf;
	}
	
	/**
	 * 
	 */
	public double[][] predictValues(DataPoint[][] x) {
		int row = x.length;
		int col = this.ulabels.length;
		double[][] pv = new double[row][col];
		double[] w = null;
		for(int i = 0; i < col; i++) {
			System.out.println("weight " + i);
			w = SparseVector.sparseVectorToArray(this.w[i], this.prob.n);
			for(int j = 0; j < row; j++) {
				pv[j][i] = SparseVector.innerProduct(w, x[j]);
			}
		}
		return pv;
	}
	
	/**
	 * 从文件读取权值 
	 * @throws IOException 
	 */
	public double[][] predictValues(DataPoint[][] x, String wfile, int col) throws IOException {
		int row = x.length;
		double[][] pv = new double[row][col];
		double[] w = null;
		
		BufferedReader in = new BufferedReader(
				new InputStreamReader(new FileInputStream(wfile)));
		String line = null;
		String[] splits = null;
		for(int i = 0; i < col; i++) {
			System.out.println("weight " + i);
			line = in.readLine();
			splits = line.split(":|\\s+|\r|\n|\t");
			w = new double[this.prob.n];
			for(int j = 0; j < splits.length / 2; j++) {
				int index = Integer.parseInt(splits[2 * j]);
				double value = Double.parseDouble(splits[2 * j + 1]);
				w[index - 1] = value;
			}
			
			for(int j = 0; j < row; j++) {
				pv[j][i] = SparseVector.innerProduct(w, x[j]);
			}
		}
		in.close();
		return pv;
	}
	
	/**
	 * 从文件读取权值 
	 * @throws IOException 
	 */
	public double[][] predictValues(DataPoint[][] x, String wfile) throws IOException {
		int row = x.length;
		int col = this.ulabels.length;
		double[][] pv = new double[row][col];
		double[] w = null;
		
		BufferedReader in = new BufferedReader(
				new InputStreamReader(new FileInputStream(wfile)));
		String line = null;
		String[] splits = null;
		for(int i = 0; i < col; i++) {
			System.out.println("weight " + i);
			line = in.readLine();
			splits = line.split(":|\\s+|\r|\n|\t");
			w = new double[this.prob.n];
			for(int j = 0; j < splits.length / 2; j++) {
				int index = Integer.parseInt(splits[2 * j]);
				double value = Double.parseDouble(splits[2 * j + 1]);
				w[index - 1] = value;
			}
			
			for(int j = 0; j < row; j++) {
				pv[j][i] = SparseVector.innerProduct(w, x[j]);
			}
		}
		in.close();
		return pv;
	}
	
	/**
	 * 单类标预测
	 */
	public int[] predictMax(DataPoint[][] x) {
		double[][] pv = predictValues(x);
		
		int[] pl = new int[pv.length];
		for(int i = 0; i < pv.length; i++) {
			System.out.println("predict " + i);
			int index = -1;
			double max = Double.NEGATIVE_INFINITY;
			for(int j = 0; j < pv[i].length; j++) {
				if(max < pv[i][j]) {
					max = pv[i][j];
					index = j;
				}
			}
			
			pl[i] = this.ulabels[index];
		}
		return pl;
	}
	
	/**
	 * 单类标预测
	 * @throws IOException 
	 */
	public int[] predictMax(DataPoint[][] x, String wfile) throws IOException {
		double[][] pv = predictValues(x, wfile);
		
		int[] pl = new int[pv.length];
		for(int i = 0; i < pv.length; i++) {
			System.out.println("predict " + i);
			int index = -1;
			double max = Double.NEGATIVE_INFINITY;
			for(int j = 0; j < pv[i].length; j++) {
				if(max < pv[i][j]) {
					max = pv[i][j];
					index = j;
				}
			}
			
			pl[i] = this.ulabels[index];
		}
		return pl;
	}
	
	/**
	 * 
	 */
	public double[][] binaryPv(double[][] pv) {
		double[][] bpv = new double[pv.length][];
		for(int i = 0; i < pv.length; i++) {
			bpv[i] = new double[pv[i].length];
			for(int j = 0; j < pv[i].length; j++) {
				if(pv[i][j] > 0) {
					bpv[i][j] = 1;
				} else {
					bpv[i][j] = -1;
				}
			}
		}
		return bpv;
	}
	
	/**
	 * 
	 */
	public int[][] predict(DataPoint[][] x) {
		double[][] pv = predictValues(x);
		int[][] pre = new int[x.length][];
		for(int i = 0; i < pv.length; i++) {
			double[] temp = pv[i];
			int counter = 0;
			for(int j = 0; j < temp.length; j++) {
				if(temp[j] > 0) {
					counter = counter + 1;
				}
			}
			
			pre[i] = new int[counter];
			counter = 0;
			for(int j = 0; j < temp.length; j++) {
				if(temp[j] > 0) {
					pre[i][counter++] = this.ulabels[j];
				}
			}
		}
		return pre;
	}
	
	/**
	 * 
	 */
	public double[] train(Problem prob, Parameter param, int k) {
		train(prob,param);
		double[][] pv = predictValues(prob.x);
		double[][] bpv = binaryPv(pv);
		bpv = pv;
		
		int[][] pre = new int[bpv.length][1];
		for(int i = 0; i < bpv.length; i++) {
			double[] x = bpv[i];
			pre[i][0] = getKnnLabels(bpv, x, prob.y, k, 1);
		}
		
		double microf1 = Measures.microf1(this.ulabels, prob.y, pre);
		double macrof1 = Measures.macrof1(this.ulabels, prob.y, pre);
		double hammingloss = Measures.averageSymLoss(prob.y, pre);
		double[] perf = {microf1, macrof1, hammingloss};
		return perf;
	}
	
	/**
	 * @throws IOException 
	 * 
	 */
	public int train(Problem prob, Parameter param, int[] k, String wfile) throws IOException {
		train(prob, param, wfile);
		double[][] pv = predictValues(prob.x, wfile);
		
		int[][][] pre = new int[k.length][pv.length][];
		for(int i = 0; i < pv.length; i++) {
			double[] x = pv[i];
			int[][] pl = getKnnLabels(pv, x, prob.y, k, 1);
			for(int j = 0; j < k.length; j++) {
				pre[j][i] = pl[j];
			}
		}
		
		double bestPerf = Double.POSITIVE_INFINITY;
		int index = -1;
		for(int i = 0; i < k.length; i++) {
			double hammingloss = Measures.averageSymLoss(prob.y, pre[i]);
			if(hammingloss < bestPerf) {
				bestPerf = hammingloss;
				index = i;
			}
		}
		return k[index];
	}
	
	/**
	 * @throws IOException 
	 * 
	 */
	public int leaveOneOut(Problem prob, int[] k, String wfile) throws IOException {
		double[][] pv = predictValues(prob.x, wfile);
		
		int[][][] pre = new int[k.length][pv.length][];
		for(int i = 0; i < pv.length; i++) {
			double[] x = pv[i];
			int[][] pl = getKnnLabels(pv, x, prob.y, k, 1);
			for(int j = 0; j < k.length; j++) {
				pre[j][i] = pl[j];
			}
		}
		
		double bestPerf = Double.POSITIVE_INFINITY;
		int index = -1;
		for(int i = 0; i < k.length; i++) {
			double hammingloss = Measures.averageSymLoss(prob.y, pre[i]);
			if(hammingloss < bestPerf) {
				bestPerf = hammingloss;
				index = i;
			}
		}
		return k[index];
	}
	
	/**
	 * 
	 */
	public int[][] predictKnnLabels(Problem train, DataPoint[][] test, int k) {
		double[][] pv = predictValues(train.x);
		double[][] bpv = binaryPv(pv);
		bpv = pv;
		
		double[][] tpv = predictValues(test);
		double[][] btpv = binaryPv(tpv);
		btpv = tpv;
		
		int[][] pre = new int[test.length][1];
		for(int i = 0; i < test.length; i++) {
			pre[i][0] = getKnnLabels(bpv, btpv[i], train.y, k, 0);
		}
		return pre;
	}
	
	/**
	 * 
	 */
	public int[][] predictKnnLabels(double[][] pv, int[][] y, double[][] tpv, int k) {		
		int[][] pre = new int[tpv.length][1];
		for(int i = 0; i < tpv.length; i++) {
			pre[i][0] = getKnnLabels(pv, tpv[i], y, k, 0);
		}
		return pre;
	}
	
	/**
	 * 
	 */
	public int predictKnnLabels(double[][] pv, int[][] y, double[] tpv, int k) {		
		int pre;
		pre = getKnnLabels(pv, tpv, y, k, 0);
		return pre;
	}
	
	/**
	 * 
	 */
	public int[] predictKnnLabels(double[][] pv, int[][] y, double[] tpv, int[] k) {		
		int[] pre = null;
		pre = getknnLabels(pv, tpv, y, k, 0);
		return pre;
	}
	
	/**
	 * @throws IOException 
	 * 
	 */
	public int[][] predictKnnLabels(Problem train, DataPoint[][] test, int k, String wfile) throws IOException {
		double[][] pv = predictValues(train.x, wfile);
		double[][] tpv = predictValues(test, wfile);
		
		int[][] pre = new int[test.length][1];
		for(int i = 0; i < test.length; i++) {
			pre[i][0] = getKnnLabels(pv, tpv[i], train.y, k, 0);
		}
		return pre;
	}
	
	/**
	 * 
	 */
	public int[][] getKnnLabels(double[][] s, double[] x, int[][] y, int[] k, int base) {
		double[] dis = new double[s.length];
		for(int i = 0; i < s.length; i++) {
			double[] sub = SparseVector.subVector(s[i], x);
			double inp = 0;
			for(int j = 0; j < sub.length; j++) {
				inp += Math.abs(sub[j]);
			}
			dis[i] = inp;
		}
		
		int[] index = Sort.getIndexBeforeSort(dis);
		
		int[][] pre = new int[k.length][];
		for(int i = 0; i < pre.length; i++) {
			pre[i] = getLabels(y, index, k[i], base);
		}
		return pre;
	}
	
	/**
	 * 
	 */
	public int[] getLabels(int[][] y, int[] index, int k, int base) {
		Map<Integer,Integer> map = new HashMap<Integer, Integer>();
		int key;
		int value;
		for(int i = base; i < base + k; i++) {
			int[] ty = y[index[i]];
			for(int j = 0; j < ty.length; j++) {
				key = ty[j];
				if(map.containsKey(key)) {
					value = map.get(key);
					value = value + 1;
					map.put(key, value);
				} else {
					map.put(key, 1);
				}
			}
		}
		
		List<Integer> list = new ArrayList<Integer>();
		Set<Integer> set = map.keySet();
		Iterator<Integer> it = set.iterator();
		while(it.hasNext()) {
			key = it.next();
			value = map.get(key);
			if(value > (double)k / 2) {
				list.add(key);
			}
		}
		
		int[] pre = new int[list.size()];
		for(int i = 0; i < pre.length; i++) {
			pre[i] = list.get(i);
		}
		return pre;
	}
	
	public int getKnnLabels(double[][] s, double[] x, int[][] y, int k, int base) {
		double[] dis = new double[s.length];
		for(int i = 0; i < s.length; i++) {
			double[] sub = SparseVector.subVector(s[i], x);
			double inp = 0;
			for(int j = 0; j < sub.length; j++) {
				inp += Math.abs(sub[j]);
			}
			dis[i] = inp;
		}
		
		int[] index = Sort.getIndexBeforeSort(dis);
		
		int[][] nl = new int[k][];
		int counter = 0;
		for(int i = base; i < base + k; i++) {
			nl[counter++] = y[index[i]];
		}
		
		int pre = multiClass(nl);
		return pre;
	}

	public int[] getknnLabels(double[][] s, double[] x, int[][] y, int[] k, int base) {
		double[] dis = new double[s.length];
		for(int i = 0; i < s.length; i++) {
			double[] sub = SparseVector.subVector(s[i], x);
			double inp = SparseVector.innerProduct(sub, sub);
			dis[i] = Math.pow(inp, 0.5);
		}
		
		int[] index = Sort.getIndexBeforeSort(dis);
		int[] pre = new int[k.length];
		
		for(int i = 0; i < k.length; i++) {
			int[][] nl = new int[k[i]][];
			int counter = 0;
			for(int j = base; j < base + k[i]; j++) {
				nl[counter++] = y[index[j]];
			}
			
			int pl = multiClass(nl);
			if(pl == -1) {
				pl = predict(x);
			}
			pre[i] = pl;
		}
		return pre;
	}
	
	/**
	 * y其实为一列向量
	 */
	public int multiClass(int[][] y) {
		int[] ys = new int[y.length];
		int[] ys_count = new int[y.length];
		int pointer = -1;
		boolean contain = false;
		
		for(int i = 0; i < y.length; i++) {
			int ty = y[i][0];
			contain = false;
			for(int j = 0; j <= pointer; j++) {
				if(ty == ys[j]) {
					ys_count[j]++;
					contain = true;
				}
			}
			
			if(!contain) {
				ys[++pointer] = ty;
				ys_count[pointer] = 1;
			}
		}
		
		int max = Integer.MIN_VALUE;
		int index = -1;
		for(int i = 0; i <= pointer; i++) {
			if(ys_count[i] > max) {
				max = ys_count[i];
				index = i;
			}
		}
		
		Arrays.sort(ys_count);
		int label = -1;
		if(ys_count.length == 1 || ys_count[ys_count.length - 1] > ys_count[ys_count.length - 2]) {
			label = ys[index];
		} 
		return label;
	}
	
	/**
	 * 
	 */
	public int sigleLabel(int[][] y) {
		Map<Integer, Integer> map = new HashMap<Integer, Integer>();
		int key;
		int value;
		for(int i = 0; i < y.length; i++) {
			for(int j = 0; j < y[i].length; j++) {
				key = y[i][j];
				if(map.containsKey(key)) {
					value = map.get(key);
					value = value + 1;
					map.put(key, value);
				} else {
					map.put(key, 1);
				}
			}
		}
		
		Set<Integer> keySet = map.keySet();
		Iterator<Integer> it = keySet.iterator();
		int max = Integer.MIN_VALUE;
		int flabel = 0;
		while(it.hasNext()) {
			key = it.next();
			value = map.get(key);
			
			if(value > max) {
				max = value;
				flabel = key;
			}
		}
		return flabel;
	}
	
	public int[] getUlabels() {
		return ulabels;
	}

	public void setUlabels(int[] ulabels) {
		this.ulabels = ulabels;
	}
	
	public void crossValidation(Problem prob, Parameter param, int n_fold, int[] k) {
		int n = prob.l;
		
		int[][][] pre = new int[k.length][n][];
		
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
			
			train(train, param);
			
			double[][] trainpv = predictValues(train.x);
			double[][] btrainpv = binaryPv(trainpv);
			btrainpv = trainpv;
			
			double[][] validpv = predictValues(valid.x);
			double[][] bvalidpv = binaryPv(validpv);
			bvalidpv = validpv;
			
			for(int ki = 0; ki < k.length; ki++) {
				
				int[][] predictLabel = new int[valid.l][1];
				for(int m = 0; m < predictLabel.length; m++) {
					predictLabel[m][0] = getKnnLabels(btrainpv, bvalidpv[m], train.y, k[ki], 0);
				}
				
				for(int j = 0; j < validIndex.length; j++) {
					pre[ki][validIndex[j]] = predictLabel[j];
				}
			}
		}
		
		for(int m = 0; m < k.length; m++) {
			double microf1 = Measures.microf1(this.ulabels, prob.y, pre[m]);
			double macrof1 = Measures.macrof1(this.ulabels, prob.y, pre[m]);
			double hammingloss = Measures.averageSymLoss(prob.y, pre[m]);
			System.out.println("c = " + param.getC() + ", k = " + k[m] + ", Micro-F1 = " + microf1 + ", Macro-F1 = " 
					+ macrof1 + ", Hamming Loss = " + hammingloss);
		}
	}

	public DataPoint[][] getW() {
		return w;
	}

	public void setW(DataPoint[][] w) {
		this.w = w;
	}

	public Structure getTree() {
		return tree;
	}

	public void setTree(Structure tree) {
		this.pathes = tree.getAllPath();
		this.maxTreeDis = tree.getMaxDistance();
		System.out.println("Max tree distance " + this.maxTreeDis);
		this.tree = tree;
	}
	
	/**
	 * 
	 */
	public void transLabels(int[][] y, Map<Integer, Integer> map) {
		for(int i = 0; i < y.length; i++) {
			for(int j = 0; j < y[i].length; j++) {
				y[i][j] = map.get(y[i][j]);
			}
		}
	}
	
	/**
	 * 
	 */
	public void getClassCenter(double[][] pv, int[][] y) {
		double[][] centers = new double[this.ulabels.length][pv[0].length];
		double[]   classCount = new double[this.ulabels.length];
		
		for(int i = 0; i < pv.length; i++) {
			System.out.println("centers " + i);
			for(int j = 0; j < this.ulabels.length; j++) {
				if(y[i][0] == this.ulabels[j]) {
					classCount[j]++;
					SparseVector.vectorAdd(centers[j], pv[i]);
				}
			}
		}
		
		for(int i = 0; i < centers.length; i++) {
			SparseVector.localVecScale(centers[i], 1 / classCount[i]);
		}
		this.classCenter = centers;
	}
	
	/**
	 * 
	 */
	public int[] predict(double[][] pv) {
		int[] labels = new int[pv.length];
		double[] dis = new double[this.ulabels.length];
		double min;
		int index;
		for(int i = 0; i < pv.length; i++) {
			System.out.print("predict " + i);
			long start = System.currentTimeMillis();
			for(int j = 0; j < dis.length; j++) {
				dis[j] = distance(pv[i], this.classCenter[j]);
			}
			
			index = -1;
			min = Double.POSITIVE_INFINITY;
			for(int j = 0; j < dis.length; j++) {
				if(dis[j] < min) {
					min = dis[j];
					index = j;
				}
			}
			
			labels[i] = this.ulabels[index];
			
			long end = System.currentTimeMillis();
			System.out.println(", " + (end - start) + "ms");
		}
		
		return labels;
	}
	
	/**
	 * 
	 */
	public int predict(double[] pv) {
		int labels;
		double max = Double.NEGATIVE_INFINITY;
		int index = -1;
		
		for(int i = 0; i < this.ulabels.length; i++) {
			if(pv[i] > max) {
				max = pv[i];
				index = i;
			}
		}
		labels = this.ulabels[index];
		return labels;
	}
	
	/**
	 * 
	 */
	public double distance(double[] a, double[] b) {
		double[] sub = SparseVector.subVector(a, b);
		double inp = SparseVector.innerProduct(sub, sub);
		return Math.pow(inp, 0.5);
	}
	
	/**
	 * 
	 */
	public double gaussian(double x, double mean, double std) {
		double it1 = 2 * Math.PI;
		it1 = Math.pow(it1, 0.5);
		it1 *= std;
		it1 = 1 / it1;
		double it2 = Math.exp(-((x - mean) * (x - mean)) / (2 * std * std));
		return it1 * it2;
	}
	
	/**
	 * 
	 */
	public void sigmoid(double[][] xs) {
		for(int i = 0; i < xs.length; i++) {
			for(int j = 0; j < xs[i].length; j++) {
				xs[i][j] = sigmoid(xs[i][j]);
			}
		}
	}
	
	/**
	 * 
	 */
	public void getMeanStd(double[][] pv, int[][] y) {
		double[][] m = new double[this.ulabels.length][];
		double[][] s = new double[this.ulabels.length][];
		
		List<double[]> list = new ArrayList<double[]>();
		for(int i = 0; i < this.ulabels.length; i++) {
			list.clear();
			for(int j = 0; j < y.length; j++) {
				if(y[j][0] == this.ulabels[i]) {
					list.add(pv[j]);
				}
			}
			
			double[][] mat = new double[list.size()][];
			for(int j = 0; j < list.size(); j++) {
				mat[j] = list.get(j);
			}
			
			double[] mean = getMean(mat);
			double[] std = getStd(mat, mean);
			m[i] = mean;
			s[i] = std;
		}
		
		this.means = m;
		this.stds = s;
	}
	
	/**
	 * 
	 */
	public double[] getMean(double[][] mat) {
		double[] sum = new double[mat[0].length];
		for(int i = 0; i < mat.length; i++) {
			sum = SparseVector.addVector(sum, mat[i]);
		}
		
		SparseVector.localVecScale(sum, 1.0 / mat.length);
		return sum;
	}
	
	/**
	 * 
	 */
	public double[] getStd(double[][] mat, double[] mean) {
		double[][] sub = new double[mat.length][];
		for(int i = 0; i < mat.length; i++) {
			sub[i] = SparseVector.subVector(mat[i], mean);
		}
		
		
		for(int i = 0; i < sub.length; i++) {
			for(int j = 0; j < sub[i].length; j++) {
				sub[i][j] = sub[i][j] * sub[i][j];
			}
		}
		
		double[] sum = new double[mat[0].length];
		for(int i = 0; i < sub.length; i++) {
			sum = SparseVector.addVector(sum, sub[i]);
		}
		
		double n = Math.max(mat.length - 1, 1);
		SparseVector.localVecScale(sum, 1.0 / n);
		
		for(int i = 0; i < sum.length; i++) {
			sum[i] = Math.pow(sum[i], 0.5);
		}
		return sum;
	}
	
	/**
	 * 
	 */
	public int[] bayesPredict(double[][] pv) {
		int[] labels = new int[pv.length];
		double[] prob = new double[this.ulabels.length];
		double max;
		int index;
		for(int i = 0; i < pv.length; i++) {
			System.out.print("predict " + i);
			long start = System.currentTimeMillis();
			
			for(int j = 0; j < this.ulabels.length; j++) {
				prob[j] = getProb(pv[i], this.means[j], this.stds[j]);
			}
			
			max = Double.NEGATIVE_INFINITY;
			index = -1;
			for(int j = 0; j < this.ulabels.length; j++) {
				if(prob[j] > max) {
					max = prob[j];
					index = j;
				}
			}
			labels[i] = this.ulabels[index];
			long end = System.currentTimeMillis();
			System.out.println(", " + (end - start) + "ms");
		}
		
		return labels;
	}
	
	/**
	 * 
	 */
	public double getProb(double[] x, double[] mean, double[] std) {
		double prob = 1.0;
		for(int i = 0; i < x.length; i++) {
			prob *= gaussian(x[i], mean[i], std[i]);
		}
		return prob;
	}
	
	/**
	 * 
	 */
	public Problem trans(double[][] pv, int dim) {
		Problem prob = new Problem();
		prob.l = pv.length;
		prob.n = pv[0].length + 1;
		prob.bias = 1;
		prob.x = new DataPoint[prob.l][];
		prob.y = new int[prob.l][];
		
		for(int i = 0; i < pv.length; i++) {
			int[] index = Sort.getIndexBeforeSort(pv[i]);
			prob.x[i] = new DataPoint[dim + 1];
			int counter = 0;
			double inp = 0;
			for(int j = index.length - 1; j >= index.length - dim; j--) {
				prob.x[i][counter++] = new DataPoint(index[j] + 1, pv[i][index[j]]);
				inp += pv[i][index[j]] * pv[i][index[j]];
			}
			inp = Math.pow(inp, 0.5);
			for(int j = 0; j < prob.x[i].length - 1; j++) {
				prob.x[i][j].value /= inp;
			}
			
			prob.x[i][counter] = new DataPoint(prob.n, 1.0);
		}
		return prob;
	}
		
	/**
	 * 两层支持向量机，第二层为当前路径到根节点路径值
	 * @throws IOException 
	 */
	public void stack(Problem prob, Parameter param1, Parameter param2, Structure tree,
			String wfile1, String wfile2) throws IOException {
		int[] nodes = tree.levelTraverse();
		
		DataPoint[] weight = null;
		
		BufferedWriter out = new BufferedWriter(
				new OutputStreamWriter(new FileOutputStream(wfile1)));
		String line = null;
		for(int i = 0; i < nodes.length; i++) {
			int label = nodes[i];
			int[] labels = constructLabels(prob.y, label);
			double[] tloss = new double[1];
			System.out.print(i + ", " + label + ", ");
			long start = System.currentTimeMillis();
			weight = Linear.train(prob, labels, param1, null, tloss, null, 0);
		
			line = new String();
			for(int j = 0; j < weight.length; j++) {
				line = line + weight[j].index + ":"  + weight[j].value + " ";
			}
			line = line + "\n";
			out.write(line);
			long end = System.currentTimeMillis();
			System.out.println((end - start) + "ms");
		}
		out.close();
		
		
		double[][] trainpv = predictValues(prob.x, wfile1, nodes.length);
//		sigmoid(trainpv);
		
		int[] leaves = tree.getLeaves();
		DataPoint[] weight1 = null;
		
		BufferedWriter out1 = new BufferedWriter(
				new OutputStreamWriter(new FileOutputStream(wfile2)));
		line = null;
		for(int i = 0; i < leaves.length; i++) {
			int label = leaves[i];
			int[] labels = constructLabels(prob.y, label);
			double[] tloss = new double[1];
			System.out.print(i + ", " + label + ", ");
			long start = System.currentTimeMillis();
			
			Problem nprob = trans(trainpv, tree, label);
			weight1 = Linear.train(nprob, labels, param2, null, tloss, null, 0);
		
			line = new String();
			for(int j = 0; j < weight1.length; j++) {
				line = line + weight1[j].index + ":"  + weight1[j].value + " ";
			}
			line = line + "\n";
			out1.write(line);
			long end = System.currentTimeMillis();
			System.out.println((end - start) + "ms");
		}
		out1.close();
	}
	
	/**
	 * @throws IOException 
	 * 
	 */
	public void stackSecondLayer(Problem prob, Parameter param1, Parameter param2, Structure tree,
			String wfile1, String wfile2) throws IOException {
		
		int[] nodes = tree.levelTraverse();	
		double[][] trainpv = predictValues(prob.x, wfile1, nodes.length);
//		sigmoid(trainpv);
		
		int[] leaves = tree.getLeaves();
		DataPoint[] weight1 = null;
		
		BufferedWriter out1 = new BufferedWriter(
				new OutputStreamWriter(new FileOutputStream(wfile2)));
		String line = null;
		for(int i = 0; i < leaves.length; i++) {
			int label = leaves[i];
			int[] labels = constructLabels(prob.y, label);
			double[] tloss = new double[1];
			System.out.print(i + ", " + label + ", ");
			long start = System.currentTimeMillis();
			
//			Problem nprob = trans(trainpv, tree, label);          //到根节点路径
			Problem nprob = trans(trainpv, tree, label);          //根节点路径和树根
			weight1 = Linear.train(nprob, labels, param2, null, tloss, null, 0);
		
			line = new String();
			for(int j = 0; j < weight1.length; j++) {
				line = line + weight1[j].index + ":"  + weight1[j].value + " ";
			}
			line = line + "\n";
			out1.write(line);
			long end = System.currentTimeMillis();
			System.out.println((end - start) + "ms");
		}
		out1.close();
	}
	
	/**
	 * @throws IOException 
	 * 
	 */
	public int[] stackPredict(DataPoint[][] xs, String wfile1, String wfile2, Structure tree) throws IOException {
		double[][] pv = predictValues(xs, wfile1, tree.levelTraverse().length);
//		sigmoid(pv);
				
		int[] leaves = tree.getLeaves();
		
		double[][] fpv = new double[xs.length][leaves.length];
		
		double[] w = null;
		
		BufferedReader in = new BufferedReader(
				new InputStreamReader(new FileInputStream(wfile2)));
		String line = null;
		String[] splits = null;
		for(int i = 0; i < leaves.length; i++) {
			int label = leaves[i];
			
			Problem nprob = trans(pv, tree, label);
			
			System.out.println("weight " + i);
			line = in.readLine();
			splits = line.split(":|\\s+|\r|\n|\t");
			w = new double[nprob.n];
			for(int j = 0; j < splits.length / 2; j++) {
				int index = Integer.parseInt(splits[2 * j]);
				double value = Double.parseDouble(splits[2 * j + 1]);
				w[index - 1] = value;
			}
			
			for(int j = 0; j < xs.length; j++) {
				fpv[j][i] = SparseVector.innerProduct(w, nprob.x[j]);
			}
		}
		in.close();
		
		int[] pl = getLabels(fpv, leaves);
		return pl;
	}
	
	/**
	 * 
	 */
	public int[] getLabels(double[][] pv, int[] labels) {
		int[] pl = new int[pv.length];
		double max = Double.NEGATIVE_INFINITY;
		int index = -1;
		for(int i = 0; i < pv.length; i++) {
			max = Double.NEGATIVE_INFINITY;
			index = -1;
			for(int j = 0; j < pv[i].length; j++) {
				if(pv[i][j] > max) {
					max = pv[i][j];
					index = j;
				}
			}
			pl[i] = labels[index];
		}
		return pl;
	}
	
	/**
	 *	全矩阵转换为DataPoint形式 
	 */
	public Problem trans(double[][] pv, Structure tree, int label) {
		Problem nprob = new Problem();
		nprob.bias = 1;
		nprob.l = pv.length;
		nprob.n = pv[0].length + 1;
		nprob.x = new DataPoint[nprob.l][];
		nprob.y = new int[nprob.l][];
		
		DataPoint[][] result = new DataPoint[pv.length][];
		
		int[] allNodes = tree.levelTraverse();
		int[] path = tree.getPathToRoot(label);
		int[] roots = tree.getChildren(tree.getRoot());
		
		Set<Integer> set = new HashSet<Integer>();
		
		for(int i = 0; i < path.length; i++) {
			if(path[i] != tree.getRoot()) {
				set.add(path[i]);
			}
		}
		
		for(int i = 0; i < roots.length; i++) {
			set.add(roots[i]);
		}
		
		int counter = set.size();
		
		int[] ids = new int[counter];
		counter = 0;
		for(int i = 0; i < allNodes.length; i++) {
			if(set.contains(allNodes[i])) {
				ids[counter++] = i;
			}
		}
		
		int index;
		double value;
		double inp = 0;
		for(int i = 0; i < pv.length; i++) {
			result[i] = new DataPoint[ids.length + 1];
			counter = 0;
			inp = 0;
			for(int j = 0; j < ids.length; j++) {
				index = ids[j] + 1;
				value = pv[i][index - 1];
				inp += value * value;
				result[i][counter++] = new DataPoint(index, value); 
			}
			
			inp = Math.pow(inp, 0.5);
			for(int j = 0; j < ids.length; j++) {
				result[i][j].value /= inp;
			}
			
			result[i][counter] = new DataPoint(pv[0].length + 1, 1);
		}
		
		nprob.x = result;
		return nprob;
	}
	
	/**
	 * 预测，权值包含中间节点，预测只在叶节点 
	 * @throws IOException 
	 */
	public int[] predictWithInnerNode(DataPoint[][] x, String wfile, Structure tree, int dim) throws IOException {
		int[] nodes = tree.levelTraverse();
		int[] leaves = tree.getLeaves();
		
		double[][] pv = new double[x.length][nodes.length];
		double[] w = null;
		
		BufferedReader in = new BufferedReader(
				new InputStreamReader(new FileInputStream(wfile)));
		String line = null;
		String[] splits = null;
		for(int i = 0; i < nodes.length; i++) {
			System.out.println("weight " + i);
			line = in.readLine();
			splits = line.split(":|\\s+|\r|\n|\t");
			w = new double[dim];
			for(int j = 0; j < splits.length / 2; j++) {
				int index = Integer.parseInt(splits[2 * j]);
				double value = Double.parseDouble(splits[2 * j + 1]);
				w[index - 1] = value;
			}
			
			for(int j = 0; j < x.length; j++) {
				pv[j][i] = SparseVector.innerProduct(w, x[j]);
			}
		}
		in.close();
		
		int[] pl = new int[pv.length];
		int[] index = getIndex(nodes, leaves);
		double max = Double.NEGATIVE_INFINITY;
		int ind = -1;
		for(int i = 0; i < pv.length; i++) {
			max = Double.NEGATIVE_INFINITY;
			ind = -1;
			for(int j = 0; j < index.length; j++) {
				if(pv[i][index[j]] > max) {
					max = pv[i][index[j]];
					ind = index[j];
				}
			}
			pl[i] = nodes[ind];
		}
		return pl;
	}
	
	/**
	 * @throws IOException 
	 * 
	 */
	public void secondLayer(DataPoint[][] x, Parameter param1, String wfile, Structure tree, int dim, String wfile1) throws IOException {
		int[] nodes = tree.levelTraverse();
		int[] leaves = tree.getLeaves();
		
		double[][] pv = new double[x.length][nodes.length];
		double[] w = null;
		
		BufferedReader in = new BufferedReader(
				new InputStreamReader(new FileInputStream(wfile)));
		String line = null;
		String[] splits = null;
		for(int i = 0; i < nodes.length; i++) {
			System.out.println("weight " + i);
			line = in.readLine();
			splits = line.split(":|\\s+|\r|\n|\t");
			w = new double[dim];
			for(int j = 0; j < splits.length / 2; j++) {
				int index = Integer.parseInt(splits[2 * j]);
				double value = Double.parseDouble(splits[2 * j + 1]);
				w[index - 1] = value;
			}
			
			for(int j = 0; j < x.length; j++) {
				pv[j][i] = SparseVector.innerProduct(w, x[j]);
			}
		}
		in.close();
		
		DataPoint[] weight = null;
		
		BufferedWriter out = new BufferedWriter(
				new OutputStreamWriter(new FileOutputStream(wfile1)));
		line = null;
		for(int i = 0; i < leaves.length; i++) {
			int label = leaves[i];
			int[] labels = getBinaryLabels(prob.y, label);
			double[] tloss = new double[1];
			System.out.print(i + ", " + label + ", ");
			long start = System.currentTimeMillis();
			
			Problem nprob = constructProblem(pv, tree, label);
			weight = Linear.train(nprob, labels, param1, null, tloss, null, 0);
		
			line = new String();
			for(int j = 0; j < weight.length; j++) {
				line = line + weight[j].index + ":"  + weight[j].value + " ";
			}
			line = line + "\n";
			out.write(line);
			long end = System.currentTimeMillis();
			System.out.println((end - start) + "ms");
		}
		out.close();
		
	}
	
	/**
	 * 
	 */
	public Problem constructProblem(double[][] pv, Structure tree, int label) {
		Problem nprob = new Problem();
		nprob.l = pv.length;
		nprob.n = tree.getAllNodes().length + 1;
		nprob.bias = 1;
		nprob.x = new DataPoint[nprob.l][];
		nprob.y = new int[nprob.l][];
		
		int[] path = tree.getPathToRoot(label);
		int[] allNodes = tree.levelTraverse();
		int root = tree.getRoot();
		
		int[] index = getIndex(allNodes, path, root);
		
		int ind;
		double value;
		int counter = 0;
		for(int i = 0; i < pv.length; i++) {
			counter = 0;
			nprob.x[i] = new DataPoint[index.length + 1];
			for(int j = 0; j < index.length; j++) {
				ind = index[j] + 1;
				value = pv[i][index[j]];
				nprob.x[i][counter++] = new DataPoint(ind, value);
			}
			
			ind = pv[0].length + 1;
			value = 1.0;
			nprob.x[i][counter] = new DataPoint(ind, value);
		}
		return nprob;
	}
	
	/**
	 * 
	 */
	public int[] getIndex(int[] allNodes, int[] path, int root) {
		int[] index = new int[path.length - 1];
		int counter = 0;
		for(int i = 0; i < path.length; i++) {
			if(path[i] != root) {
				for(int j = 0; j < allNodes.length; j++) {
					if(path[i] == allNodes[j]) {
						index[counter++] = j;
					}
				}
			}
		}
		return index;
	}
	
	/**
	 * 
	 */
	public int[] getIndex(int[] allNodes, int[] leaves) {
		int[] index = new int[leaves.length];
		for(int i = 0; i < leaves.length; i++) {
			index[i] = getInd(allNodes, leaves[i]);
		}
		return index;
	}
	
	/**
	 * 
	 */
	public int getInd(int[] allNodes, int node) {
		int index = -1;
		for(int i = 0; i < allNodes.length; i++) {
			if(allNodes[i] == node) {
				index = i;
				break;
			}
		}
		return index;
	}
	
	/**
	 * 
	 */
	public double entropyThreshold(double[] ent, int num) {
		int[] index = Sort.getIndexBeforeSort(ent);
		double cut = 0;
		if(ent.length <= num) {
			return cut;
		}
		
		int ind = ent.length - 1 - num;
		return ent[index[ind]];
	}
	
	/**
	 * 
	 */
	public double[] delta(double[][] tpv, int[][] y, double epoch) {
		int[] labels = this.ulabels;
		double[] ds = new double[labels.length];
		for(int i = 0; i < ds.length; i++) {
			int label = labels[i];
			for(int j = 0; j < tpv.length; j++) {
				if(label == y[j][0]) {
					double max = com.tools.Matrix.findVecMax(tpv[j]);
					double correct = tpv[j][i];
					double sub = max - correct;
					if(sub > ds[i]) {
						ds[i] = sub;
					}
				}
			}
		}
		
//		double max = Double.NEGATIVE_INFINITY;
//		for(int i = 0; i < ds.length; i++){
//			if(ds[i] > max) {
//				max = ds[i];
//			}
//		}
		for(int i = 0; i < ds.length; i++) {
			ds[i] = ds[i] / epoch;
		}
		return ds;
	}
	
	/**
	 * 
	 */
	public double[][] getT(double[][] tpv, int[][] y, double epoch) {
		double[] ds = delta(tpv, y, (int)epoch);
		int[] labels = this.ulabels;
		double[][] ts = new double[100][labels.length];
		double[] cs = new double[labels.length]; 
				
		int counter = 1;
		while(counter++ < 100) {
			for(int i = 0; i < tpv.length; i++) {
				double[] pv = Matrix.vecAdd(tpv[i], cs);
				int index = Matrix.findLabel(labels, y[i][0]);
				double max = Matrix.findVecMax(pv);
				if(pv[index] < max) {
					cs[index] += ds[index];
				}
			}
			
			for(int i = 0; i < labels.length; i++) {
				ts[counter - 1][i] = cs[i];
			}
		}
		return ts;
	}
	
	public double[][] predictValues(DataPoint[][] x, double[] t, String wfile) throws IOException {
		int row = x.length;
		int col = this.ulabels.length;
		double[][] pv = new double[row][col];
		double[] w = null;
		
		BufferedReader in = new BufferedReader(
				new InputStreamReader(new FileInputStream(wfile)));
		String line = null;
		String[] splits = null;
		for(int i = 0; i < col; i++) {
			System.out.println("weight " + i);
			line = in.readLine();
			splits = line.split(":|\\s+|\r|\n|\t");
			w = new double[this.prob.n];
			for(int j = 0; j < splits.length / 2; j++) {
				int index = Integer.parseInt(splits[2 * j]);
				double value = Double.parseDouble(splits[2 * j + 1]);
				w[index - 1] = value;
			}
			
			for(int j = 0; j < row; j++) {
				pv[j][i] = SparseVector.innerProduct(w, x[j]);
			}
		}
		
		for(int i = 0; i < pv.length; i++) {
			pv[i] = Matrix.vecAdd(pv[i], t);
		}
		in.close();
		return pv;
	}
	
	/**
	 * 
	 */
	public int[] predictMax(double[][] pv) {
		int[] labels = new int[pv.length];
		double max;
		int index;
		for(int i = 0; i < pv.length; i++) {
			System.out.print("predict " + i);
			long start = System.currentTimeMillis();
			index = -1;
			max = Double.NEGATIVE_INFINITY;
			for(int j = 0; j < pv[i].length; j++) {
				if(pv[i][j] > max) {
					max = pv[i][j];
					index = j;
				}
			}
			labels[i] = this.ulabels[index];
			
			long end = System.currentTimeMillis();
			System.out.println(", " + (end - start) + "ms");
		}
		
		return labels;
	}
	
	public double sigmoid(double d) {
		return 1.0 / (1.0 + Math.exp(-d));
	}
	
	public double[] sigmoid(double[] v) {
		double[] r = new double[v.length];
		for(int i = 0; i < v.length; i++) {
			r[i] = sigmoid(v[i]);
		}
		return r;
	}
	
	public double[] subVec(double[] a, double[] b) {
		if(a.length != b.length) {
			return null;
		}
		
		double[] s = new double[a.length];
		for(int i = 0; i < s.length; i++) {
			s[i] = a[i] - b[i];
		}
		return s;
	}
	
	public double[] eleMulti(double[] a, double[] b) {
		if(a.length != b.length) {
			return null;
		}
		
		double[] m = new double[a.length];
		for(int i = 0; i < m.length; i++) {
			m[i] = a[i] * b[i];
		}
		return m;
	}
	
	public double[] subScaleVec(double s, double[] v) {
		double[] r = new double[v.length];
		for(int i = 0; i < r.length; i++) {
			r[i] = s - v[i];
		}
		return r;
	}
	
	public double[] addVec(double[] a, double[] b) {
		if(a.length != b.length) {
			return null;
		}
		double[] sum = new double[a.length];
		for(int i = 0; i < sum.length; i++) {
			sum[i] = a[i] + b[i];
		}
		return sum;
	}
	
	/**
	 * 
	 */
	public double[] getVecy(int[] labels, int label) {
		double[] y = new double[labels.length];
		for(int i = 0; i < labels.length; i++) {
			if(labels[i] == label) {
				y[i] = 1.0;
			} else {
				y[i] = 0.0;
			}
		}
		return y;
	}
	
	/**
	 * 
	 */
	public double[] getCost(int[] labels, int label) {
		double[] cost = new double[labels.length];
		for(int i = 0; i < labels.length; i++) {
			if(labels[i] == label) {
				cost[i] = 5;
			} else {
				cost[i] = 1.0;
			}
		}
		return cost;
	}
	
	public void vecScale(double[] v, double s) {
		for(int i = 0; i < v.length; i++) {
			v[i] *= s;
		}
	}
	
	public double[][] getTs(double[][] pv, int[][] y, double lr, int epoch) {
		double[] t = new double[this.ulabels.length];
		for(int i = 0; i < t.length; i++) {
			t[i] = (Math.random() * 2) - 1;
		}
		
		double[][] rt = new double[epoch + 1][this.ulabels.length];
		
		int counter = 0;
		while(counter++ < epoch) {
			int[] rs = RandomSequence.randomSequence(pv.length);
			for(int i = 0; i < pv.length; i++) {
				double[] finalpv = addVec(pv[rs[i]], t);
				double[] ty = getVecy(this.ulabels, y[rs[i]][0]);
				double[] it1 = subVec(finalpv, ty);
				double[] it2 = subScaleVec(1, finalpv);
				double[] cost = getCost(this.ulabels, y[rs[i]][0]);
				cost = eleMulti(cost, cost);
				it2 = eleMulti(finalpv, it2);
				double[] del = eleMulti(it1, it2);
				del = eleMulti(del, cost);
				vecScale(del, 2.0 * lr);
				t = subVec(t, del);
			}
			
			double obj = 0;
			for(int i = 0; i < pv.length; i++) {
				double[] ty = getVecy(this.ulabels, y[i][0]);
				double[] fiv = addVec(pv[i], t);
				fiv = sigmoid(fiv);
				double[] s = subVec(fiv, ty);
				obj += innerProduct(s, s);
			}
			System.out.println("obj = " + obj);
			for(int i = 0; i < this.ulabels.length; i++) {
				rt[counter][i] = t[i];
			}
		}
		return rt;
	}
	
	
	public double[][] getTs(double[][] pv, int[][] y, double lr, double lambda, int epoch) {
		double[] t = new double[this.ulabels.length];
		for(int i = 0; i < t.length; i++) {
			t[i] = Math.random();
		}
		
		double[][] rt = new double[epoch + 1][this.ulabels.length];
		
		int counter = 0;
		while(counter++ < epoch) {
			int[] rs = RandomSequence.randomSequence(pv.length);
			for(int i = 0; i < pv.length; i++) {
				double[] finalpv = addVec(pv[rs[i]], t);
				double[] ty = getVecy(this.ulabels, y[rs[i]][0]);
				double[] it1 = subVec(finalpv, ty);
				double[] it2 = subScaleVec(1, finalpv);
				it2 = eleMulti(finalpv, it2);
				double[] del = eleMulti(it1, it2);
				double[] nt = scale(t, lambda);
				del = addVec(del, nt);
				vecScale(del, 2.0 * lr);
				t = subVec(t, del);
			}
			
			double obj = 0;
			for(int i = 0; i < pv.length; i++) {
				double[] ty = getVecy(this.ulabels, y[i][0]);
				double[] fiv = addVec(pv[i], t);
				fiv = sigmoid(fiv);
				double[] s = subVec(fiv, ty);
				obj += innerProduct(s, s);
			}
			double obj1 = lambda * innerProduct(t, t);
			obj += obj1;
			System.out.println("obj = " + obj);
			for(int i = 0; i < this.ulabels.length; i++) {
				rt[counter][i] = t[i];
			}
		}
		return rt;
	}
	
	public double[] scale(double[] vec, double s) {
		double[] r = new double[vec.length];
		for(int i = 0; i < r.length; i++) {
			r[i] = vec[i] * s;
		}
		return r;
	}
	
	public double innerProduct(double[] a, double[] b) {
		double sum = 0;
		for(int i = 0; i < a.length; i++) {
			sum += a[i] * b[i];
		}
		return sum;
	}
	
	public double[][] getTPbpu(double[][] pv, int[][] y, int epoch) {
		double[][] ts = new double[epoch][this.ulabels.length];
		double[] t = new double[this.ulabels.length];
		
		int counter = 0;
		double accuracy = Double.NEGATIVE_INFINITY;
		double ft = 0;
		while(counter++ < epoch) {
			int[] rs = RandomSequence.randomSequence(this.ulabels.length);
			for(int i = 0; i < this.ulabels.length; i++) {
				int oldi = i;
				
				i = rs[i];
				int label = this.ulabels[i];
				long start = System.currentTimeMillis();
				System.out.print("i = " + i + ", ");
				double[] r = getRanges(pv, y, i, label);
				
				accuracy = Double.NEGATIVE_INFINITY;
				ft = 0;
				
				for(int j = 0; j < r.length - 1; j++) {
					t[i] = (r[j] + r[j + 1]) / 2.0;
					int[] pl = predict(pv, t);
					double tac = accuracy(y, pl);
					if(tac > accuracy) {
						accuracy = tac;
						ft = t[i];
					}
				}
				t[i] = ft;
				long end = System.currentTimeMillis();
				System.out.println((end - start) + "ms" + ", t[" + i + "] = " + t[i] + ", accuracy = " + accuracy);
				
				i = oldi;
			}
			for(int i = 0; i < this.ulabels.length; i++) {
				ts[counter-1][i] = t[i];
			}
		}
		return ts;
	}
	
	
	public double[][] getTPbpu(double[][] pv, int[][] y, double[][] tpv, int[][] ty, int epoch) {
		double[][] ts = new double[epoch][this.ulabels.length];
		double[] t = new double[this.ulabels.length];
		
		int counter = 0;
		double accuracy = Double.NEGATIVE_INFINITY;
		double ft = 0;
		while(counter++ < epoch) {
			int[] rs = RandomSequence.randomSequence(this.ulabels.length);
			for(int i = 0; i < this.ulabels.length; i++) {
				int oldi = i;
				
				i = rs[i];
				int label = this.ulabels[i];
				long start = System.currentTimeMillis();
				System.out.print("i = " + i + ", ");
				double[] r = getRanges(pv, y, i, label);
				
				accuracy = Double.NEGATIVE_INFINITY;
				ft = 0;
				
				for(int j = 0; j < r.length - 1; j++) {
					t[i] = (r[j] + r[j + 1]) / 2.0;
					int[] pl = predict(pv, t);
					double tac = accuracy(y, pl);
					if(tac > accuracy) {
						accuracy = tac;
						ft = t[i];
					}
				}
				t[i] = ft;
				long end = System.currentTimeMillis();
				
				int[] tpl = predict(tpv, t);
				double ac = accuracy(ty, tpl);
				
				System.out.println((end - start) + "ms" + ", t[" + i + "] = " + t[i] + ", accuracy = " + accuracy + ", test accuracy = " + ac);
				
				i = oldi;
			}
			System.out.println();
			for(int i = 0; i < this.ulabels.length; i++) {
				ts[counter-1][i] = t[i];
			}
		}
		return ts;
	}
	
	public int[] predict(double[][] pv, double[] t) {
		int[] pl = new int[pv.length];
		double max = Double.NEGATIVE_INFINITY;
		int index = -1;
		for(int i = 0; i < pv.length; i++) {
			double[] fpv = addVec(pv[i], t);
			max = Double.NEGATIVE_INFINITY;
			index = -1;
			for(int j = 0; j < fpv.length; j++) {
				if(fpv[j] > max) {
					max = fpv[j];
					index = j;
				}
			}
			pl[i] = this.ulabels[index];
		}
		return pl;
	}
	
	public double accuracy(int[][] y, int[] pl) {
		double counter = 0;
		for(int i = 0; i < y.length; i++) {
			if(y[i][0] == pl[i]) {
				counter++;
			}
		}
		double accuracy = counter / y.length;
		return accuracy;
	}
	
	public double[] getRanges(double[][] pv, int[][] y, int col, int label) {
		double[] rs = new double[pv.length];
		int tl;
		for(int i = 0; i < pv.length; i++) {
			if(Contain.contain(y[i], label)) {
				tl = 1;
			} else {
				tl = 0;
			}
			
			int mi = findMaxIndex(pv[i]);
			
			if(tl == 1 && mi == col) {
				double second = findMaxExcludeCol(pv[i], col);
				rs[i] = second - pv[i][col];
			} else if(tl == 1 && mi != col) {
				rs[i] = pv[i][mi] - pv[i][col];
			} else if(tl == 0 && mi == col) {
				double second = findMaxExcludeCol(pv[i], col);
				rs[i] = second - pv[i][col];
			} else if(tl == 0 && mi != col) {
				rs[i] = pv[i][mi] - pv[i][col];
			}
		}
		
		Arrays.sort(rs);
		return rs;
	}
	
	/**
	 * 
	 */
	public int findMaxIndex(double[] vec) {
		int index = -1;
		double max = Double.NEGATIVE_INFINITY;
		for(int i = 0; i < vec.length; i++) {
			if(vec[i] > max) {
				max = vec[i];
				index = i;
			}
		}
		return index;
	}
	
	public double findMaxExcludeCol(double[] vec, int col) {
		double max = Double.NEGATIVE_INFINITY;
		for(int i = 0; i < vec.length; i++) {
			if(i == col) {
				continue;
			} else {
				if(vec[i] > max) {
					max = vec[i];
				}
			}
		}
		return max;
	}
	
	public int findMaxIndexExcludeCol(double[] vec, int col) {
		double max = Double.NEGATIVE_INFINITY;
		int index = -1;
		for(int i = 0; i < vec.length; i++) {
			if(i == col) {
				continue;
			}
			
			if(vec[i] > max) {
				max = vec[i];
				index = i;
			}
		}
		return index;
	}
	
	public int[] findRunnerUp(double[][] pv, double[] t, int col) {
		int[] index = new int[pv.length];
		for(int i = 0; i < pv.length; i++) {
			double[] tpv = addVec(pv[i], t);
			index[i] = findMaxIndexExcludeCol(tpv, col);
		}
		return index;
	}
	
	public double[] addMatrixCol(double[][] pv, double con, int col) {
		double[] r = new double[pv.length];
		for(int i = 0; i < pv.length; i++) {
			r[i] = pv[i][col] + con;
		}
		return r;
	}
	
	public int[] predict(double[] colValue, double[] otherMax, int currentCol, int[] otherCol) {
		int[] pl = new int[colValue.length];
		for(int i = 0; i < pl.length; i++) {
			if(colValue[i] > otherMax[i]) {
				pl[i] = this.ulabels[currentCol];
			} else {
				pl[i] = this.ulabels[otherCol[i]];
			}
		}
		return pl;
	}
	
	public double[][] getTPbpuV1(double[][] pv, int[][] y, int epoch) {
		double[][] ts = new double[epoch][this.ulabels.length];
		double[] t = new double[this.ulabels.length];
		
		int counter = 0;
		double accuracy = Double.NEGATIVE_INFINITY;
		double ft = 0;

		while(counter++ < epoch) {
			int[] rs = RandomSequence.randomSequence(this.ulabels.length);
			for(int i = 0; i < this.ulabels.length; i++) {
				int oldi = i;
				
				i = rs[i];
				int label = this.ulabels[i];
				long start = System.currentTimeMillis();
				System.out.print("i = " + i + ", ");
				double[] r = getRanges(pv, y, i, label);
				
				accuracy = Double.NEGATIVE_INFINITY;
				ft = 0;
				
				int[] runnerup = findRunnerUp(pv, t, i);
				double[] secondMaxValue = getValues(pv, runnerup);

				for(int j = 0; j < r.length - 1; j++) {
					t[i] = (r[j] + r[j + 1]) / 2.0;
					double[] colValue = addMatrixCol(pv, t[i], i);
					int[] pl = predict(colValue, secondMaxValue, i, runnerup);
					double tac = accuracy(y, pl);
					if(tac > accuracy) {
						accuracy = tac;
						ft = t[i];
					}
				}
				t[i] = ft;
				long end = System.currentTimeMillis();
				System.out.println((end - start) + "ms" + ", t[" + i + "] = " + t[i] + ", accuracy = " + accuracy);
			
				i = oldi;
			}
			
			for(int i = 0; i < this.ulabels.length; i++) {
				ts[counter-1][i] = t[i];
			}
		}
		return ts;
	}
	
	public double[] getValues(double[][] pv, int[] index) {
		double[] vs = new double[pv.length];
		for(int i = 0; i < pv.length; i++) {
			vs[i] = pv[i][index[i]];
		}
		return vs;
	}
	
	public int[] getLabels(int[] index) {
		int[] pl = new int[index.length];
		for(int i = 0; i < pl.length; i++) {
			pl[i] = this.ulabels[index[i]];
		}
		return pl;
	}
	
	/**
	 * 
	 */
	public double getBestT(double[][] pv, double[] t, int col, int[][] y) {
		int[] labels = this.ulabels;
		int[] maxIndex = findMaxIndex(pv, t);
		int[] secondIndex = findRunnerUp(pv, t, col);
		int[] ro = getRealBooleanLabel(y, col);              //真实类别0/1
		int[] pl = getPredictLabels(maxIndex);
		int[] plo = getBooleanPredict(y, pl);				//预测正误0/1
		double[] range = getRange(pv, maxIndex, secondIndex);

		double[] copyRange = Arrays.copyOf(range, range.length);
		Arrays.sort(copyRange);
		
		int err = 0;
		int cor = 0;
		
		int diff = Integer.MIN_VALUE;
		
		for(int i = 0; i < range.length - 1; i++) {
			double tt = (copyRange[i] + copyRange[i+1]) / 2;
			for(int j = 0; j < range.length; j++) {
				if(ro[j] == 1 && plo[j] == 1 && range[j] < tt) {
					err++;
				}
				
				if(ro[j] == 1 && plo[j] == 0 && range[j] < tt) {
					cor++;
				}
				
				if(ro[j] == 0 && plo[j] == 0 && range[j] < tt) {
					err++;
				}
				
				if(ro[j] == 0 && plo[j] ==1 
						&& this.ulabels[secondIndex[j]] == y[j][0]
								&& tt < range[j]) {
					err--;
				}
			}
		}
		return 0;
	}
	
	public double[] getRange(double[][] pv, int[] maxIndex, int[] secondIndex) {
		double[] r = new double[pv.length];
		for(int i = 0; i < r.length; i++) {
			r[i] = pv[i][secondIndex[i]] - pv[i][maxIndex[i]];
		}
		return r;
	}
	
	public int[] getRealBooleanLabel(int[][] y, int col) {
		int[] bl = new int[y.length];
		int label = this.ulabels[col];
		for(int i = 0; i < bl.length; i++) {
			if(y[i][0] == label) {
				bl[i] = 1;
			} else {
				bl[i] = 0;
			}
		}
		return bl;
	}
	
	public int[] getPredictLabels(int[] maxIndex) {
		int[] pl = new int[maxIndex.length];
		for(int i = 0; i < pl.length; i++) {
			pl[i] = this.ulabels[maxIndex[i]];
		}
		return pl;
	}
	
	public int[] getBooleanPredict(int[][] y, int[] pl) {
		int[] bl = new int[y.length];
		for(int i = 0; i < bl.length; i++) {
			if(y[i][0] == pl[i]) {
				bl[i] = 1;
			} else {
				bl[i] = 0;
			}
		}
		return bl;
	}
	
	/**
	 * 
	 */
	public int[] findMaxIndex(double[][] pv, double[] t) {
		int[] index = new int[pv.length];
		double max = Double.NEGATIVE_INFINITY;
		int ind = -1;
		for(int i = 0; i < pv.length; i++) {
			max = Double.NEGATIVE_INFINITY;
			ind = -1;
			for(int j = 0; j < pv[i].length; j++) {
				double[] sum = addVec(pv[i], t);
				if(sum[j] > max) {
					max = sum[j];
					ind = j;
				}
			}
			index[i] =ind;
		}
		return index;
	}
	
	/**
	 * 
	 */
	public double[][] ann(double[][] xs, int[][] y, double lr, int epoch, double[][] tpv, int[][] ty) {
		int[] labels = this.ulabels;
		double[][] w = new double[labels.length][labels.length];
		for(int i = 0; i < w.length; i++) {
			w[i][i] = 1.0;
		}
		
		int tc = 0;
		while(tc++ < epoch) {
			int[] rs = RandomSequence.randomSequence(xs.length);
			for(int i = 0; i < rs.length; i++) {
				double[] x = xs[rs[i]];
				double[] output = getOutput(x, w);
				double[] target = getVecy(labels, y[rs[i]][0]);
				double[] diff = subVec(output, target);
				diff = eleMulti(diff, output);
				diff = eleMulti(diff, subScaleVec(1.0, output));
				diff = scale(diff, -2 * lr);
				double[][] deltaw = outputProduct(diff, x);
				matAdd(w, deltaw);
			}
			
			int[] trainpl = annPredict(xs, w);
			int[] testpl = annPredict(tpv, w);
			double counter = 0;
			for(int i = 0; i < trainpl.length; i++) {
				if(trainpl[i] == y[i][0]) {
					counter++;
				}
			}
			double trainAccuracy = counter / trainpl.length;
			
			counter = 0;
			for(int i = 0; i < testpl.length; i++) {
				if(testpl[i] == ty[i][0]) {
					counter++;
				}
			}
			double testAccuracy = counter / testpl.length;
			
			double sum = 0;
			for(int i = 0; i < rs.length; i++) {
				double[] x = xs[i];
				double[] output = getOutput(x, w);
				double[] target = getVecy(labels, y[i][0]);
				double[] diff = subVec(output, target);
				sum += innerProduct(diff, diff);
			}
			System.out.println("obj = " + sum + ", train accuracy " + trainAccuracy
					+ ", test accuracy = " + testAccuracy);
		}
		return w;
	}
	
	public void matAdd(double[][] w, double[][] deltaw) {
		for(int i = 0; i < w.length; i++) {
			for(int j = 0; j < w[i].length; j++) {
				w[i][j] += deltaw[i][j];
			}
		}
	}
	
	public double[][] outputProduct(double[] row, double[] col) {
		double[][] result = new double[row.length][col.length];
		for(int i = 0; i < row.length; i++) {
			for(int j = 0; j < col.length; j++) {
				result[j][i] = row[i] * col[j];
			}
		}
		return result;
	}
	
	public double[] getOutput(double[] x, double[][] w) {
		double[] r = new double[w.length];
		double sum = 0;
		for(int i = 0; i < w.length; i++) {
			sum = 0;
			for(int j = 0; j < w.length; j++) {
				sum += x[j] * w[j][i];
			}
			r[i] = sigmoid(sum);
		}
		return r;
	}
	
	public int[] annPredict(double[][] pv, double[][] w) {
		int[] pl = new int[pv.length];
		int index = -1;
		double max = Double.NEGATIVE_INFINITY;
		for(int i = 0; i < pv.length; i++) {
			double[] outi = getOutput(pv[i], w);
			index = -1;
			max = Double.NEGATIVE_INFINITY;
			for(int j = 0; j < this.ulabels.length; j++) {
				if(outi[j] > max) {
					max = outi[j];
					index = j;
				}
			}
			pl[i] = this.ulabels[index];
		}
		return pl;
	}
	
	//(a1w1 + a2w2 + ... + anwn) ^ 2 + c * sum(si)
	public DataPoint[] optimize(DataPoint[][] w, DataPoint[][] x, int[] y, int dim, double c, double precision
			, int epoch) {
		double[] a = new double[w.length];
		double[] alpha = new double[x.length];
		double lastObj = getObejctFunc(a, w, x, y, c);
		
		double[][] fullW = new double[w.length][dim];
		for(int i = 0; i < fullW.length; i++) {
			fullW[i] = SparseVector.sparseVectorToArray(w[i], dim);
		}
		
		double[][] tFullW = Matrix.trans(fullW);
		double[][] invWW = Matrix.inv(Matrix.multi(fullW, tFullW));
		
		double[][] M = Matrix.multi(Matrix.multi(tFullW, invWW), fullW);
		
	
		DataPoint[] ayx = null;
		int count = 0;
		while(true) {
			int[] index = RandomSequence.randomSequence(x.length);
			for(int i = 0; i < index.length; i++) {
				double alpha_old = alpha[index[i]];
				DataPoint[] tx = x[index[i]];
				int ty = y[index[i]];
				tx = SparseVector.slVector(tx, ty);
				
				double[] mayx = new double[dim];
				if(ayx != null) {
					mayx = Matrix.multi(M, ayx);
				}
				
				double mayxyx = Matrix.multi(mayx, tx);
				double[] myx = Matrix.multi(M, tx);
				double myxayx = 0;
				if(ayx != null) {
					myxayx = Matrix.multi(myx, ayx);
				}
				double mayxmyx = SparseVector.innerProduct(mayx, myx);
				
				double top = mayxyx + myxayx - mayxmyx - 1;
				double myxmyx = SparseVector.innerProduct(myx, myx);
				double myxyx = Matrix.multi(myx, tx);
				double bottom = myxmyx - 2 * myxyx;
				double d = top / bottom;
				
				double alpha_new = Math.max(0, Math.min(alpha_old + d, c));
				alpha[index[i]] = alpha_new;
				
				double delta = alpha_new - alpha_old;
				SparseVector.scaleVector(tx, delta);
				ayx = SparseVector.addVector(ayx, tx);
			}
			
			double[] wayx = Matrix.multi(fullW, ayx);
			a = Matrix.multi(invWW, wayx);
			
			double primalObj = getObejctFunc(a, w, x, y, c);
			double dualObj = getDualObject(M, ayx, alpha);
			
//			System.out.print("Primal Obj = " + primalObj + ", Dual Obj = " + dualObj + ", ");
//			for(int i = 0; i < a.length; i++) {
//				System.out.print(a[i] + " ");
//			}
//			System.out.println();
			count++;
			System.out.println(count);
			if(Math.abs(primalObj - dualObj) < precision
					|| count > epoch) {
				break;
			}
		}
		DataPoint[] rv = null;
		for(int i = 0; i < a.length; i++) {
			DataPoint[] tw = SparseVector.slVector(w[i], a[i]);
			rv = SparseVector.addVector(rv, tw);
		}
		return rv;
	}
	
	/**
	 * 获得对偶函数目标值
	 */
	public double getDualObject(double[][] M, DataPoint[] ayx, double[] alpha) {
		double sumAlpha = 0.0;
		for(int i = 0; i < alpha.length; i++) {
			sumAlpha += alpha[i];
		}
		
		double[] mayx = Matrix.multi(M, ayx);
		double mayx2 = 0.5 * SparseVector.innerProduct(mayx, mayx);
		double mayxayx = Matrix.multi(mayx, ayx);
		double obj = mayx2 + sumAlpha - mayxayx;
		return obj;
	}
	
	public double[] predictRealValue(double[] as, DataPoint[][] w, DataPoint[][] x, int[] y) {
		DataPoint[] weight = null;
		for(int i = 0; i < w.length; i++) {
			DataPoint[] tw = SparseVector.slVector(w[i], as[i]);
			weight = SparseVector.addVector(weight, tw);
		}
		double[] pv = new double[x.length];
		for(int i = 0; i < pv.length; i++) {
			pv[i] = SparseVector.innerProduct(weight, x[i]);
		}
		return pv;
	}
	
	public double kerc(double[] as, DataPoint[][] w, DataPoint[][] x, int[] y) {
		double[] pv = predictRealValue(as, w, x, y);
		double sumk = 0.0;
		for(int i = 0; i < pv.length; i++) {
			sumk += Math.max(0, 1 - y[i] * pv[i]);
		}
		return sumk;
	}
	
	public double weightNorm(double[] as, DataPoint[][] w) {
		DataPoint[] weight = null;
		for(int i = 0; i < w.length; i++) {
			DataPoint[] tw = SparseVector.slVector(w[i], as[i]);
			weight = SparseVector.addVector(weight, tw);
		}
		return SparseVector.innerProduct(weight,  weight);
	}
	
	public double getObejctFunc(double[] as, DataPoint[][] w, DataPoint[][] x, int[] y, double c) {
		double wn = 0.5 * weightNorm(as, w);
		double kerc = c * kerc(as, w, x, y);
		return (wn + kerc);
	}
	
	/**
	 * 第一层训练权值
	 */
	public DataPoint[][] getFirstWeight(Problem prob, Parameter param, Structure tree) {
		DataPoint[][] weight = new DataPoint[tree.getAllNodes().length][];
		int[] nodes = tree.levelTraverse();
		for(int i = 0; i < nodes.length; i++) {
			int id = nodes[i];
			int[] labels = constructLabels(prob.y, id);
			double[] tloss = new double[1];
			weight[id] = Linear.train(prob, labels, param, null, tloss, null, 0);		
		}
		return weight;
	}
	
	public DataPoint[][] getSecondWeight(Problem prob, double c, double precision, Structure tree,
			DataPoint[][] firstWeight, int dim, boolean leafOnly, int epoch) {
		DataPoint[][] secondWeight = new DataPoint[tree.getAllNodes().length][];
		int[] nodes = null;
		if(leafOnly) {
			nodes = tree.getLeaves();
		} else {
			nodes = tree.levelTraverse();
		}
		
		for(int i = 0; i < nodes.length; i++) {
			int id = nodes[i];
			int[] path = tree.getPathToRoot(id);
			if(path.length <= 2) {
				continue;
			}
			int[] y = constructLabels(prob.y, id);
			
			DataPoint[][] tw = new DataPoint[path.length - 1][];
			int index = 0;
			for(int j  = 0; j < path.length; j++) {
				if(path[j] != tree.getRoot()) {
					tw[index++] = firstWeight[path[j]];
				}
			}
			secondWeight[id] = optimize(tw, prob.x, y, dim, c, precision, epoch);
//			System.out.println();
		}
		return secondWeight;
	}
	
	/**
	 * 第二层为逻辑回归
	 * */
	public DataPoint[][] getLRSecondWeight(Problem prob, Structure tree, DataPoint[][] firstWeight, 
			int dim, boolean leafOnly, double lr, double precision, int epoch) {
		DataPoint[][] secondWeight = new DataPoint[tree.getAllNodes().length][];
		int[] nodes = null;
		if(leafOnly) {
			nodes = tree.getLeaves();
		} else {
			nodes = tree.levelTraverse();
		}
		
		for(int i = 0; i < nodes.length; i++) {
			int id = nodes[i];
			int[] path = tree.getPathToRoot(id);
			if(path.length <= 2) {
				continue;
			}
			int[] y = constructLabels(prob.y, id);
			
			DataPoint[][] tw = new DataPoint[path.length - 1][];
			int index = 0;
			for(int j  = 0; j < path.length; j++) {
				if(path[j] != tree.getRoot()) {
					tw[index++] = firstWeight[path[j]];
				}
			}
			secondWeight[id] = optimizeLR(tw, prob.x, y, dim, lr, precision, epoch);
//			System.out.println();
		}
		return secondWeight;
	}
	
	
	/**
	 * Logistic regression,逻辑回归问题求解 
	 */
	public DataPoint[] optimizeLR(DataPoint[][] tw, DataPoint[][] x, int[] y, int dim, double lr, double precision, int epoch)
	{
		double[] a = new double[tw.length];
		for(int i = 0; i < a.length; i++) {
			a[i] = Math.random();
		}
		
		for(int i = 0; i < y.length; i++) {
			if(y[i] == -1) {
				y[i] = 0;
			}
		}
		
		double[][] fullW = new double[tw.length][dim];
		for(int i = 0; i < fullW.length; i++) {
			fullW[i] = SparseVector.sparseVectorToArray(tw[i], dim);
		}
		
		double lastObj = Double.NEGATIVE_INFINITY;
		double currentObj = 0;
		int count = 0;
		while(true) {
			int[] index = RandomSequence.randomSequence(x.length);
			for(int i = 0; i < index.length; i++) {
				DataPoint[] tx = x[index[i]];
				int ty = y[index[i]];
				double[] wat = multiMatVec(fullW, a);
				double watxi = SparseVector.innerProduct(wat, tx);
				double ui = sigmoidLR(watxi);
				double[] delta = multiMatSparVec(fullW, tx);
				delta = SparseVector.scaleVector(delta, lr * (ty - ui));
				a = SparseVector.addVector(a, delta);
			}
			
			currentObj = 0;
			for(int i = 0; i < x.length; i++) {
				DataPoint[] tx = x[i];
				int ty = y[i];
				double[] wat = multiMatVec(fullW, a);
				double watxi = SparseVector.innerProduct(wat, tx);
				double ui = sigmoidLR(watxi);
				currentObj += ty * (Math.log(ui)) + (1 - ty) * Math.log(1 - ui);
			}
			count++;
			
			if(Math.abs(currentObj - lastObj) / Math.abs(currentObj) <= precision || count > epoch) {
				break;
			}
			lastObj = currentObj;
		}
		
		for(int i = 0; i < a.length; i++) {
			System.out.print(a[i] + " ");
		}
		System.out.println();
		
		double[] res = multiMatVec(fullW, a);
		DataPoint[] result = SparseVector.arrayToSparseVector(res);
		return result;
		
	}
	
	
	/**
	 * 
	 */
	public double[] multiMatSparVec(double[][] mat, DataPoint[] x) {
		if(mat == null || x == null) {
			return null;
		}
		
		double[] result = new double[mat.length];
		for(int i = 0; i < result.length; i++) {
			for(int j = 0; j < x.length; j++) {
				result[i] += mat[i][x[j].index - 1] * x[j].value;
			}
		}
		return result;
	}
	
	/**
	 * 矩阵与向量相乘，与普通意义上的矩阵向量相乘不同 
	 */
	public double[] multiMatVec(double[][] mat, double[] vec) {
		if(mat == null || vec == null) {
			return null;
		}
		
		double[] result = new double[mat[0].length];
		for(int i = 0; i < mat.length; i++) {
			for(int j = 0; j < mat[i].length; j++) {
				result[j] += mat[i][j] * vec[i];
			}
		}
		return result;
	}
 	
	public double sigmoidLR(double x) 
	{
		return 1.0 / (1.0 + Math.exp(-x));
	}
	
	/**
	 * 二级预测 
	 */
	public int[][] secondLevelPredictMulti(DataPoint[][] secondWeight, DataPoint[][] x) {
		int[][] pl = new int[x.length][];
		for(int i = 0; i < x.length; i++) {
			DataPoint[] tx = x[i];
			double[] pv = new double[secondWeight.length];
			int count = 0;
			for(int j = 0;j < pv.length; j++) {
				if(secondWeight[j] == null) {
					pv[j] = Double.NEGATIVE_INFINITY;
				}
				pv[j] = SparseVector.innerProduct(secondWeight[j], tx);
				if(pv[j] > 0) {
					count++;
				}
			}
			pl[i] = new int[count];
			count = 0;
			for(int j = 0; j < pv.length; j++) {
				if(pv[j] > 0) {
					pl[i][count++] = j;
				}
			}
		}
		return pl;
	}
	
	/**
	 * 二级预测 
	 */
	public int[][] secondLevelPredictMax(DataPoint[][] secondWeight, DataPoint[][] x) {
		int[][] pl = new int[x.length][];
		for(int i = 0; i < x.length; i++) {
			DataPoint[] tx = x[i];
			double[] pv = new double[secondWeight.length];
			for(int j = 0;j < pv.length; j++) {
				if(secondWeight[j] == null) {
					pv[j] = Double.NEGATIVE_INFINITY;
				}
				pv[j] = SparseVector.innerProduct(secondWeight[j], tx);
			}
			pl[i] = new int[1];
			double max = Double.NEGATIVE_INFINITY;
			int ind = -1;
			for(int j = 0; j < pv.length; j++) {
				if(pv[j] > max) {
					max = pv[j];
					ind = j;
				}
			}
			pl[i][0] = ind;
		}
		return pl;
	}
	
	/**
	 * 二级预测 
	 */
	public int[][] secondLevelLrPredictMax(DataPoint[][] secondWeight, DataPoint[][] x) {
		int[][] pl = new int[x.length][];
		for(int i = 0; i < x.length; i++) {
			DataPoint[] tx = x[i];
			double[] pv = new double[secondWeight.length];
			for(int j = 0;j < pv.length; j++) {
				if(secondWeight[j] == null) {
					pv[j] = Double.NEGATIVE_INFINITY;
				}
				pv[j] = SparseVector.innerProduct(secondWeight[j], tx);
			}
			pl[i] = new int[1];
			double max = Double.NEGATIVE_INFINITY;
			int ind = -1;
			for(int j = 0; j < pv.length; j++) {
				if(pv[j] > max) {
					max = pv[j];
					ind = j;
				}
			}
			pl[i][0] = ind;
		}
		return pl;
	}
	
	/**
	 * 第二层为逻辑回归
	 * */
	public DataPoint[][] getHRLRSecondWeight(Problem prob, Structure tree, DataPoint[][] firstWeight, 
			int dim, boolean leafOnly, double c, double lr, double precision, int epoch) {
		DataPoint[][] secondWeight = new DataPoint[tree.getAllNodes().length][];
		int[] nodes = null;
		if(leafOnly) {
			nodes = tree.getLeaves();
		} else {
			nodes = tree.levelTraverse();
		}
		
		for(int i = 0; i < nodes.length; i++) {
			int id = nodes[i];
			int[] path = tree.getPathToRoot(id);
			if(path.length <= 2) {
				continue;
			}
			int[] y = constructLabels(prob.y, id);
			
			DataPoint[][] tw = new DataPoint[path.length - 1][];
			int index = 0;
			for(int j  = 0; j < path.length; j++) {
				if(path[j] != tree.getRoot()) {
					tw[index++] = firstWeight[path[j]];
				}
			}
			secondWeight[id] = optimizeHRLR(tw, prob.x, y, dim, c, lr, precision, epoch);
//			System.out.println();
		}
		return secondWeight;
	}
	
	public DataPoint[] optimizeHRLR(DataPoint[][] tw, DataPoint[][] x, int[] y, int dim,
			double c, double lr, double precision, int epoch){
		double[][] fullW = new double[tw.length][dim];
		for(int i = 0; i < tw.length; i++) {
			fullW[i] = SparseVector.sparseVectorToArray(tw[i], dim);
		}
		
		double[][] tfullW = Matrix.trans(fullW);
		double[][] wtw = Matrix.multi(fullW,  tfullW);
		
		double[] a = new double[tw.length];
		for(int i = 0; i < a.length; i++) {
			a[i] = Math.random();
		}
		
		double lastObj = Double.POSITIVE_INFINITY;
		int count = 0;
		while(true) {
			double[] wtwa = multiMatVec(wtw, a);
			double[] delta = new double[a.length];
			double[] wa = multiMatVec(fullW, a);
			for(int i = 0; i < x.length; i++) {
				DataPoint[] tx = x[i];
				int ty = y[i];
				double mywax = -ty * SparseVector.innerProduct(wa, tx);
				double s = (1.0 / (1.0 + Math.exp(mywax))) * Math.exp(mywax) * (-ty);
				double[] wtx = multiMatSparVec(fullW, tx);
				wtx = SparseVector.scaleVector(wtx, s);
				delta = SparseVector.addVector(delta, wtx);
			}
			double s = lr * c;
			delta = SparseVector.scaleVector(delta, s);
			a = SparseVector.subVector(a, delta);
			
			double currentObj = 0;
			wa = multiMatVec(fullW, a);
			double it1 = 0.5 * SparseVector.innerProduct(wa, wa);
			double it2 = 0;
			for(int i = 0; i < x.length; i++) {
				DataPoint[] tx = x[i];
				int ty = y[i];
				double miywax = -ty * SparseVector.innerProduct(wa, tx);
				it2 += Math.log(1.0 + Math.exp(miywax));
			}
			currentObj = it1 + c * it2;
//			System.out.println("obj = " + currentObj);
			count++;
			double d = Math.abs(Math.abs(currentObj) - Math.abs(lastObj)) / Math.abs(currentObj);
			if(count > epoch || d < precision) {
				break;
			}
			lastObj = currentObj;
		}
//		for(int i = 0; i < a.length; i++) {
//			System.out.print(a[i] + " ");
//		}
//		System.out.println();
		
		double[] wa = multiMatVec(fullW, a);
		DataPoint[] result = SparseVector.arrayToSparseVector(wa);
		return result;
	}
	
	public double[] multiMuVe(double[][] mat, double[] a) {
		double[] result = new double[mat.length];
		for(int i = 0; i < mat.length; i++) {
			for(int j = 0; j < mat[i].length; j++) {
				result[i] += mat[i][j] * a[j];
			}
		}
		return result;
	}
}
