package com.rssvm;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import com.fileInputOutput.Parameter;
import com.fileInputOutput.Problem;
import com.sparseVector.DataPoint;
import com.sparseVector.SparseVector;
import com.structure.Structure;
import com.threshold.Scutfbr;
import com.tools.Contain;
import com.tools.CrossValidation;
import com.tools.RandomSequence;
import com.tools.Statistics;



public class RecursiveSVM {
	private Structure 		structure;
	private DataPoint[][] 	weights;
	private Problem 		prob;
	private Parameter 		param;
	private double 			precision;
	private DataPoint[][] 	all_alpha;
	private int[] 			labels;
	private double[] 		thresholds;
	private Random			random;
	private DataPoint[][] 	validWeight;
	

	public RecursiveSVM(Structure structure, Problem prob, Parameter param, double precision) throws IOException {
		this.structure 	= structure;
		this.param 		= param;
		this.prob 		= prob;
		this.weights 	= new DataPoint[structure.getTotleVertex()][];
		this.precision 	= precision;
		this.validWeight = new DataPoint[structure.getTotleVertex()][];
		this.labels		= Statistics.getUniqueLabels(this.prob.y);
		this.thresholds	= new double[this.labels.length];
		this.random		= new Random();
		this.thresholds = new double[labels.length];
	}
	
	public DataPoint[][] train(Problem train, Parameter param) throws IOException {
		int[] nodes = this.structure.levelTraverse();
		this.all_alpha = new DataPoint[nodes.length][];
		double 			obj 		= 0;
		double[] 		loss 		= new double[1];
		double 			totleLoss 	= 0;
		DataPoint[][] 	w = new DataPoint[nodes.length][];
		int 		id;
		double 		delta 	= 0;
		double 		lastObj = 0;
		int 		tc 		= 0;
		while(tc < param.getMaxIteration()) {
			totleLoss 	= 0;
			obj 		= 0;			
			for(int i = 0; i < nodes.length; i++) {
				id = nodes[i];
				if(!structure.isLeaf(id)) {
					updataInnerNode(w, id);
				} else {
					updateLeafNode(w, id, train, param, loss);
					totleLoss += loss[0];
				}
			}		
			
			for(int i = 0; i < nodes.length; i++) {
				id 	= 	nodes[i];
				obj += 	getRegularTerm(w, id);
			}
			
			obj += param.getC() * totleLoss;
			
			delta = Math.abs(obj - lastObj) / lastObj;
			
			if(delta <= this.precision) {
				break;
			}
			lastObj = obj;
			tc++;
		}
		
		this.weights = w;
		return w;
	}
	
	
	public DataPoint[][] trainNew(Problem train, Parameter param) throws IOException {
		int[] nodes = this.structure.levelTraverse();
		this.all_alpha = new DataPoint[nodes.length][];
		double 			obj 		= 0;
		double[] 		loss 		= new double[1];
		double 			totleLoss 	= 0;
		DataPoint[][] 	w = new DataPoint[nodes.length][];
		int 		id;
		double 		delta 	= 0;
		double 		lastObj = 0;
		int 		tc 		= 0;
		while(tc < param.getMaxIteration()) {
			totleLoss 	= 0;
			obj 		= 0;
			
			String basefile = "weight";
			String filename = basefile + tc + ".txt";
			BufferedWriter out = new BufferedWriter(
					new OutputStreamWriter(new FileOutputStream(filename)));
			
			for(int i = 0; i < nodes.length; i++) {
				long start = System.currentTimeMillis();
				System.out.print("node " + i + ", ");
				if(!structure.isLeaf(nodes[i])) {
					updataInnerNode(w, nodes, i);
				} else {
					updateLeafNode(w, nodes, i, train, param, loss);
					totleLoss += loss[0];
					String line = new String();
					for(int j = 0; j < w[i].length; j++) {
						line += w[i][j].index + ":" + w[i][j].value + " ";
					}
					out.write(line);
				}
				long end = System.currentTimeMillis();
				System.out.println((end - start) + "ms");
			}
			out.close();
			
			for(int i = 0; i < nodes.length; i++) {
				id 	= 	nodes[i];
				obj += 	getRegularTerm(w, id);
			}
			
			obj += param.getC() * totleLoss;
			
			delta = Math.abs(obj - lastObj) / lastObj;
			
			if(delta <= this.precision) {
				break;
			}
			lastObj = obj;
			tc++;
		}
		
		this.weights = w;
		return w;
	}
	
	/**
	 * @throws IOException 
	 * 
	 */
	public int[] predictNew(DataPoint[][] x, String wfile) throws IOException {
		int[] nodes = this.structure.levelTraverse();
		int[] leaves = this.structure.getLeaves();
		List<Integer> list = new ArrayList<Integer>();
		
		for(int i = 0; i < nodes.length; i++) {
			if(Contain.contain(leaves, nodes[i])) {
				list.add(nodes[i]);
			}
		}
		
		double[][] pv = predictValues(x, wfile, leaves.length);
		
		int[] pre = new int[x.length];
		double max = Double.POSITIVE_INFINITY;
		int index = -1;
		for(int i =  0; i < pv.length; i++) {
			max = Double.POSITIVE_INFINITY;
			index = -1;
			for(int j = 0; j < pv[i].length; j++) {
				if(pv[i][j] > max) {
					max = pv[i][j];
					index = j;
				}
			}
			pre[i] = leaves[index];
		}
		return pre;
	}
	
	/**
	 * @throws IOException 
	 * 
	 */
	public void writeLabelToFile(int[] pre, String lfile) throws IOException {
		BufferedWriter out = new BufferedWriter(
				new OutputStreamWriter(new FileOutputStream(lfile)));
		for(int i = 0; i < pre.length; i++) {
			out.write(pre[i] + "\n");
		}
		out.close();
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
	 * 构造叶节点id的标签
	 * */
	public int[] getLabel(int[][] y, int id) {
		int[] result = new int[y.length];
		int[] temp;
		for(int i = 0; i < result.length; i++) {
			temp = y[i];
			result[i] = -1;
			for(int j = 0; j < temp.length; j++) {
				if(temp[j] == id) {
					result[i] = 1;
					break;
				}
			}
		}
		return result;
	}
	
	/**
	 * 更新中间节点权值
	 * */
	public void updataInnerNode(DataPoint[][] w, int id) {
		int pid = structure.getParent(id);
		int[] childId = structure.getChildren(id);
		
		double totleNieghbour = 0;
		DataPoint[] sum = null;
		if(pid != -1) {
			totleNieghbour++;
			sum = SparseVector.copyScaleVector(w[pid], 1);
		}
		
		if(childId != null && childId.length != 0) {
			for(int i = 0; i < childId.length; i++) {
				totleNieghbour++;
				sum = SparseVector.addVector(sum, w[childId[i]], prob.n);
			}
		}
		w[id] = SparseVector.copyScaleVector(sum, (1 / totleNieghbour));
		
	}

	/**
	 * 更新中间节点权值
	 * */
	public void updataInnerNode(DataPoint[][] w, int[] nodes, int j) {
		int pid = structure.getParent(nodes[j]);
		int[] childId = structure.getChildren(nodes[j]);
		
		double totleNieghbour = 0;
		DataPoint[] sum = null;
		if(pid != -1) {
			totleNieghbour++;
			sum = SparseVector.copyScaleVector(w[pid], 1);
		}
		
		if(childId != null && childId.length != 0) {
			for(int i = 0; i < childId.length; i++) {
				totleNieghbour++;
				sum = SparseVector.addVector(sum, w[childId[i]], prob.n);
			}
		}
		w[j] = SparseVector.copyScaleVector(sum, (1 / totleNieghbour));
		
	}
	
	
	/**
	 * 更新叶节点
	 * */
	public void newUpdateLeafNode(DataPoint[][] w, int id, Problem prob, Parameter param, double[] loss) {
		int[] label = null;
		label = new int[prob.l];
		for(int i = 0; i < label.length; i++) {
			if(Contain.contain(prob.y[i], id)) {
				label[i] = 1;
			} else {
				label[i] = -1;
			}
		}
		
		int[][] index = getSubProblemIndex(prob, id);
		Problem subprob = getSubProblem(prob, index);
		
//		label = new int[subprob.l];
//		for(int i = 0; i < label.length; i++) {
//			label[i] = subprob.y[i][0];
//		}
		
		int[] path = structure.getPathToRoot(id);
		
		DataPoint[] pw = null;
		if(path != null) {
			for(int i = 0; i < path.length; i++) {
				pw = SparseVector.addVector(pw, w[path[i]]);
			}
		}
		
		int pid = structure.getParent(id);
		DataPoint[] parent = null;
		if(pid != -1) {
			parent = w[pid];
		}
		
		SparseVector.scaleVector(pw,  1 / ((double)path.length));
		w[id] = Linear.train(prob, label, param, pw, loss, this.all_alpha, id);
//		w[id] = Linear.train(subprob, label, param, parent, loss, this.all_alpha, id);
//		w[id] = Linear.train(subprob, label, param, null, loss, this.all_alpha, id);
	}
	
	
	/**
	 * 获得挑选出来的子问题
	 * */
	public Problem getSubProblem(Problem prob, int[][] indexs) {
		Problem nprob = new Problem();
		nprob.bias = prob.bias;
		nprob.l = indexs[0].length;
		nprob.n = prob.n;
		nprob.x = new DataPoint[nprob.l][];
		nprob.y = new int[nprob.l][];
		
		for(int i = 0; i < nprob.l; i++) {
			nprob.x[i] = prob.x[indexs[0][i]];
			nprob.y[i] = new int[]{indexs[1][i]};
		}
		
		return nprob;
	}
	
	/**
	 * 以自身为子树的标签为正，属于父节点，不属于自身为负
	 * */
	public int[][] getSubProblemIndex(Problem prob, int id) {
		int pid = this.structure.getParent(id);
		
		int[] negative = this.structure.getDes(pid);
		int[] positive = this.structure.getDes(id);
		
		List<Integer> index = new ArrayList<Integer>();
		List<Integer> label = new ArrayList<Integer>();
		
		int i;
		int[] ty = null;
		int y = -1;
		for(i = 0; i < prob.l; i++) {
			ty = prob.y[i];
			if(Contain.subcontain(negative, ty)) {
				y = -1;
				index.add(i);
				if(Contain.subcontain(positive, ty)) {
					y = 1;
				}
				label.add(y);
			}
		}
		
		int[][] result = new int[2][index.size()];
		for(i = 0; i < index.size(); i++) {
			result[0][i] = index.get(i);
			result[1][i] = label.get(i);
		}
		return result;
	}
	
	
	/**
	 * 
	 * */
	public int[] getLabels(int[] ids, int[][] y) {
		int[] result = new int[y.length];
		boolean flag = false;
		int j;
		for(int i = 0; i < result.length; i++) {
			flag = false;
			for(j = 0; j < ids.length; j++) {
				if(Contain.contain(y[i], ids[j])) {
					flag = true;
					break;
				}
			}
			
			if(flag == true) {
				result[i] = 1;
			} else {
				result[i] = -1;
			}
		}
		
		return result;
	}
	
	/**
	 * 	更新叶节点权值 
	 * */
	public void updateLeafNode(DataPoint[][] w, int id, Problem prob, Parameter param, double[] loss) {
		int[] label = getLabels(id, prob.y);
		int pid = structure.getParent(id);
		DataPoint[] parent = null;
		if(pid != -1) {
			parent = w[pid];
		}
		w[id] = Linear.train(prob, label, param, parent, loss, this.all_alpha, id);
	}
	
	
	/**
	 * 	更新叶节点权值 
	 * */
	public void updateLeafNode(DataPoint[][] w, int[] nodes, int  i, Problem prob, Parameter param, double[] loss) {
		int[] label = getLabels(nodes[i], prob.y);
		int pid = structure.getParent(nodes[i]);
		DataPoint[] parent = null;
		if(pid != -1) {
			parent = w[pid];
		}
		w[i] = Linear.train(prob, label, param, parent, loss, this.all_alpha, i);
	}
	
	/**
	 * 求节点与父节点权值差的模
	 * */
	public double getRegularTerm(DataPoint[][] w, int id) {
		double result = 0;
		int pid = this.structure.getParent(id);
		DataPoint[] parent = null;
		if(pid != -1) {
			parent = w[pid];
		}
		DataPoint[] wid = w[id];
		DataPoint[] sub = SparseVector.subVector(wid, parent, prob.n);
		if(sub != null && sub.length != 0) {
			result = 0.5 * SparseVector.innerProduct(sub, sub, prob.n);
		} else {
			result = 0;
		}
		return result;
	}

	/**
	 * 预测单个样本类别
	 * */
	public int[] predictSingelSample(DataPoint[][] weight, DataPoint[] sample) {
		int[] leaves = this.labels;
		double predicti = 0;
		double[] predictValues = new double[leaves.length];
		int[] result = null;
		
		for(int i = 0; i < leaves.length; i++) {
			if(weight[leaves[i]] == null) {				//对权值为空的节点不计算输出。问题出在向量相乘时有一个为null返回值为0，而不是抛出异常
				predicti = -Double.MAX_VALUE;
			} else {
				predicti = SparseVector.innerProduct(weight[leaves[i]], sample, prob.n);
			}
			predictValues[i] = predicti;
		}
		
//		result = getPredict(leaves, predictValues, this.thresholds);
		if(result == null || result.length == 0) {
			result = getPredict(leaves, predictValues);
		}
		return result;
	}
	
	/**
	 * 根据叶节点输出值返回预测类标
	 * */
	public int[] getPredict(int[] labels, double[] predictValues) {
		boolean allNegative = true;
		int i;
		for(i = 0; i < predictValues.length; i++) {
			if(predictValues[i] >= 0) {
				allNegative = false;
				break;
			}
		}
		
		
		int[] result = null;
		if(allNegative) {
//System.out.println("all negative");
			int index = -1;
			double max = Double.NEGATIVE_INFINITY;
			
			for(i = 0; i < predictValues.length; i++) {
				if(predictValues[i] > max) {
					max = predictValues[i];
					index = i;
				}
			}
			
			result = new int[1];
			result[0] = labels[index];
			return result;
			
		} else {
			int counter = 0;
			for(i = 0; i < predictValues.length; i++) {
				if(predictValues[i] >= 0) {
					counter++;
				}
			}
			result = new int[counter];
			
			counter = 0;
			for(i = 0; i < predictValues.length; i++) {
				if(predictValues[i] >= 0) {
					result[counter++] = labels[i];
				}
			}
			return result;
		}
	}
	
	
	public int[][] largeScalePredict(DataPoint[][] weight, DataPoint[][] samples) {
		
		int[] labels = this.structure.getLeaves();
		double[][] result = new double[samples.length][labels.length];
		
		double[] w = null;
		
		int i, j;
		
		int id;
		DataPoint[] sample;
		for(i = 0; i < labels.length; i++) {
			id = labels[i];
			w = SparseVector.sparseVectorToArray(weight[id], this.prob.n);
			for(j = 0; j < samples.length; j++) {
				sample = samples[j];
				result[j][i] = SparseVector.innerProduct(w, sample);
			}
		}
		
		int[][] finalLabel = new int[samples.length][1];
		
		int index = -1;
		double max;
		for(i = 0; i < finalLabel.length; i++) {
			
			max = Double.NEGATIVE_INFINITY;
			for(j = 0; j < result[i].length; j++) {
				if(result[i][j] > max) {
					max = result[i][j];
					index= j;
				}
			}
			
			finalLabel[i][0] = labels[index];
		}
		return finalLabel;
	}
	
	
	public double[][] predictValues(DataPoint[][] weight, DataPoint[][] samples) {
		int[] allLabels = this.labels;
		double[][] result = new double[samples.length][allLabels.length];
		
		int i, j;
		double[][] w = new double[weight.length][];
		for(i = 0; i < weight.length; i++) {
			w[i] = SparseVector.sparseVectorToArray(weight[i], this.prob.n);
		}
		
		DataPoint[] sample = null;
		for(i = 0; i < samples.length; i++) {
			sample = samples[i];
			for(j = 0; j < allLabels.length; j++) {
				result[i][j] = SparseVector.innerProduct(w[allLabels[j]], sample);
			}
		}
		return result;
	}
	
	/**
	 *  
	 */
	public int[][] predict(DataPoint[][] weight, DataPoint[][] samples) {
		int[][] result = new int[samples.length][];
		DataPoint[] sample = null;
		double[][] w = new double[weight.length][];
		for(int i = 0; i < w.length; i++) {
			w[i] = SparseVector.sparseVectorToArray(weight[i], this.prob.n);
		}
		
		for(int i = 0; i < samples.length; i++) {
			sample = samples[i];
			result[i] = predictSingleSample(w, sample);
		}
		return result;
	}
	
	/**
	 * 	预测输出，输出为正输出1，输出为负输出-1。
	 * */
	public int[] predictSingleSample(double[][] weight, DataPoint[] sample) {
		int[] las = this.labels;
		double[] preval = new double[las.length];
		int i;
		for(i = 0; i < las.length; i++) {
			preval[i] = SparseVector.innerProduct(weight[las[i]], sample);
		}
		
//		int counter = 0;
//		for(i = 0; i < preval.length; i++) {
//			if(preval[i] > 0) {
//				counter++;
//			}
//		}
		
		int index = 0;
		double max = Double.NEGATIVE_INFINITY;
		for(i = 0; i < preval.length; i++) {
			if(preval[i] > max) {
				index = i;
				max = preval[i];
			}
		}
		
//		int[] result = new int[counter];
//		counter = 0;
//		for(i = 0; i < preval.length; i++) {
//			if(preval[i] > 0) {
//				result[counter++] = las[i];
//			}
//		}
		
		int[] result = new int[1];
		result[0] = las[index];
		return result;
	}
	
	/**
	 * 
	 * */
	public int[] getLabels(int id, int[][] y) {

		
		int[] result = new int[y.length];
		int[] ty;
		for(int i = 0; i < result.length; i++) {
			ty = y[i];
			if(Contain.contain(ty, id)) {
				result[i] = 1;
			} else {
				result[i] = -1;
			}
			
		}
		return result;
	}
	
	//
	public boolean numInArr(int[] arr, int num) {
		boolean result = false;
		for(int i = 0; i < arr.length; i++) {
			if(num == arr[i]) {
				result = true;
				break;
			}
		}
		return result;
	}

	public Structure getStructure() {
		return structure;
	}

	public void setStructure(Structure structure) {
		this.structure = structure;
	}

	public DataPoint[][] getWeights() {
		return weights;
	}

	public void setWeights(DataPoint[][] weights) {
		this.weights = weights;
	}

	public Problem getProb() {
		return prob;
	}

	public void setProb(Problem prob) {
		this.prob = prob;
	}

	public Parameter getParam() {
		return param;
	}

	public void setParam(Parameter param) {
		this.param = param;
	}

	public double getPrecision() {
		return precision;
	}

	public void setPrecision(double precision) {
		this.precision = precision;
	}
	
	/**
	 * 所有样本标签是否都为-1
	 * */
	public int numOfPositiveSamples(int[] y) {
		int counter = 0;
		for(int i = 0; i < y.length; i++) {
			if(y[i] == 1) {
				counter++;
			}
		}
		return counter;
	}
	
	
	/**
	 * 统计标签中正例的个数
	 * */
	public int getNumOfPositiveSamples(int[] labels) {
		int result = 0;
		if(labels == null) {
			return result;
		}
		for(int i = 0; i < labels.length; i++) {
			if(labels[i] == 1) {
				result++;
			}
		}
		
		return result;
	}
	
	/**
	 * 
	 */
	public double[] predictValues(DataPoint[][] weight, DataPoint[] samples) {
		double[] result = new double[this.labels.length];
		
		DataPoint[] w = null;
		double sum;
		for(int i = 0; i < result.length; i++) {
			w = weight[this.labels[i]];
			if(w == null) {
				result[i] = Double.NEGATIVE_INFINITY;
			} else {
				sum = SparseVector.innerProduct(w, samples, this.prob.n);
				result[i] = sum;
			}
		}
		return result;
	}
	
	public void swap(int[] a, int i, int j) {
		int temp = a[i];
		a[i] = a[j];
		a[j] = temp;
	}
	
	/**
	 * 	
	 * */
	public int[] getLabels(int[][] y, int id) {
		int[] result = new int[y.length];
		for(int i = 0; i < result.length; i++) {
			if(Contain.contain(y[i], id)) {
				result[i] = 1;
			} else {
				result[i] = -1;
			}
		}
		
		return result;
	}


	public DataPoint[][] getValidWeight() {
		return validWeight;
	}

	public void setValidWeight(DataPoint[][] validWeight) {
		this.validWeight = validWeight;
	}
	
	public DataPoint[][] flatTrain(Problem train, Parameter param) throws IOException {
		int[] nodes = this.structure.levelTraverse();
		
		this.all_alpha = new DataPoint[this.structure.getTotleVertex()][];
		
		double[] 		loss 		= new double[1];
		
		DataPoint[][] 	w = new DataPoint[this.structure.getTotleVertex()][];

		int 		id;
	
		for(int i = 0; i < nodes.length; i++) {
			id = nodes[i];
			updateNode(w, id, train, param);
		}
		this.weights = w;
		return w;
	}
	
	public double[] newCrossValidation(Problem prob, Parameter param, int n_fold) throws IOException {
		int n = prob.l;	
		int[] index = RandomSequence.randomSequence(n);
		
		int segLength = n / n_fold;
		
		int vbegin = 0;
		int vend = 0;		
		
		int[] validIndex = null;
		int[] trainIndex = null;
		int counter = 0;				
		System.out.print("c = " + param.getC());
		int i;
		double[] result = new double[5];
		
		for(i = 0; i < n_fold; i++) {
			
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
				
			double[] temp = trainValidate(train, valid, param);
		
			
			for(int j = 0; j < result.length; j++) {
				result[j] += temp[j];
			}
			
		}
		
		for(int j = 0; j < result.length; j++) {
			result[j] /= n_fold;
		}
		System.out.println(", Micro-F1 = " + result[0] + ", Macro-F1 = " + result[1] + ", Hamming Loss = " + result[3] + 
				", Zero One Loss = " + result[4] + ", it " + result[2]);
		return result;
	}
	
	
	/**
	 * 返回值0 microf1, 返回值1 macrof1, 返回值2 迭代次数, 返回值3 hamming loss
	 */
	public double[] trainValidate(Problem train, Problem valid, Parameter param) throws IOException {
		int[] nodes = this.structure.getAllNodes();
		
		this.all_alpha = new DataPoint[this.structure.getTotleVertex()][];
		
		double[] 		loss 		= new double[1];

		
		DataPoint[][] 	w = new DataPoint[this.structure.getTotleVertex()][];

		int 		id;	
		int 		tc 		= 0;
		
		double lastmif1 = 0;
		double lastmcf1 = 0;
		double micro_f1 = 0;
		double macro_f1 = 0;
		double zeroneloss = 0;
		double lasthl 	= Double.POSITIVE_INFINITY;
		double hammingloss = 0;
		
		while(tc < 1) {
			tc++;

			int[] rs = RandomSequence.randomSequence(nodes.length);

			for(int i = 0; i < nodes.length; i++) {
				id = nodes[rs[i]];
				if(!structure.isLeaf(id)) {
//					updataInnerNode(w, id);
				} else {
					updateLeafNode(w, id, train, param, loss);
				}
			}		
			
			int[][] pre = predict(w, valid.x);
			
			micro_f1 = Measures.microf1(this.labels, valid.y, pre);
			macro_f1 = Measures.macrof1(this.labels, valid.y, pre);
			hammingloss = Measures.averageSymLoss(valid.y, pre);
			zeroneloss = Measures.zeroOneLoss(valid.y, pre);
			
			//学习曲线，迭代退出条件
			if(hammingloss >= lasthl) {
				break;
			}
			
			this.weights = w;
			lastmif1 = micro_f1;
			lastmcf1 = macro_f1;
			lasthl   = hammingloss;
		}
		
		double[] result = new double[5];
		result[0] = lastmif1;
		result[1] = lastmcf1;
		result[2] = (tc - 1);
		result[3] = hammingloss;
		result[4] = zeroneloss;
		
		return result;
	}
	
	
	/**
	 * 不扩展中间节点，中间节点当做独立的两类分类问题
	 * */
	public DataPoint[][] newTrain(Problem train, Problem valid, Parameter param) throws IOException {
		int root = this.structure.getRoot();
		int[] leafs = this.structure.levelTraverse();
		
		this.all_alpha = new DataPoint[this.structure.getTotleVertex()][];
		
		double[] 		loss 		= new double[1];
	
		DataPoint[][] 	w = new DataPoint[this.structure.getTotleVertex()][];
		int 		id;
		int 		tc 		= 0;	
		while(tc < 1000) {		
			for(int i = 0; i < leafs.length; i++) {
				id = leafs[i];
				newUpdateLeafNode(w, id, train, param, loss);
			}
//			updataInnerNode(w, root);		

			int[][] pre = newPredict(w, valid.x);
//			pre = Consistance.fixLabels(structure, pre);
			
			double micro_f1 = Measures.microf1(leafs, valid.y, pre);
			double macro_f1 = Measures.macrof1(leafs, valid.y, pre);
System.out.println("c = " + param.getC() + ", Validate, Micro-F1 = " + micro_f1 + ", Macro-F1 = " + macro_f1);
 
			pre = newPredict(w, train.x);
//			pre = Consistance.fixLabels(structure, pre);
			micro_f1 = Measures.microf1(leafs, train.y, pre);
			macro_f1 = Measures.macrof1(leafs, train.y, pre);
System.out.println("c = " + param.getC() + ", Train, Micro-F1 = " + micro_f1 + ", Macro-F1 = " + macro_f1);
System.out.println();
			
			tc++;
		}
		this.weights = w;
		return w;
	}
	
	/**
	 * 中间节点也作为类别的预测函数，没有考虑类别一致性问题
	 * */
	public int[][] newPredict(DataPoint[][] w, DataPoint[][] samples) {
		int[] labels = this.structure.levelTraverse();
		int[][] result = new int[samples.length][];
		
		
		double[][] weight = new double[w.length][];
		int i, j;
		for(i = 0; i < weight.length; i++) {
			weight[i] = SparseVector.sparseVectorToArray(w[i], this.prob.n);
		}
		
		DataPoint[] sample = null;
		int counter = 0;
		double[] pv = null;
		for(i = 0; i < samples.length; i++) {
			sample = samples[i];
			pv = new double[labels.length];
			counter = 0;
			for(j = 0; j < labels.length; j++) {
				pv[j] = SparseVector.innerProduct(weight[labels[j]], sample);
				if(pv[j] > 0) {
					counter++;
				}
			}
			
			result[i] = new int[counter];
			counter = 0;
			for(j = 0; j < labels.length; j++) {
				if(pv[j] > 0) {
					result[i][counter++] = labels[j];
				}
			}
		}
		
		return result;
	}
	
	/**
	 * 更新每个节点，不考虑父子关系
	 * */
	public void updateNode(DataPoint[][] w, int id, Problem prob, Parameter param) {
		
		int[] label = getLabels(id, prob.y);
			
		int nops = numOfPositiveSamples(label);
		
		if(nops == 0) {
			w[id] = null;							//权值为null，导致最后计算w * x 时返回值为0。异常处理
			return;
		}

		double[] loss = new double[1];
		w[id] = Linear.train(prob, label, param, null, loss, this.all_alpha, id);
	}
	
	/**
	 *	迭代指定次数 
	 */
	public DataPoint[][] train(Problem train, Parameter param, int iteration) throws IOException {
		int[] nodes = this.structure.getAllNodes();
		this.all_alpha = new DataPoint[this.structure.getTotleVertex()][];
		double[] 		loss 		= new double[1];	
		DataPoint[][] 	w = new DataPoint[this.structure.getTotleVertex()][];
		int 		id;
		int 		tc 		= 0;
		
		while(tc < iteration) {
			int[] rs = RandomSequence.randomSequence(nodes.length);
	
			for(int i = 0; i < nodes.length; i++) {
				id = nodes[rs[i]];

				if(!structure.isLeaf(id)) {
					updataInnerNode(w, id);
				} else {
					updateLeafNode(w, id, train, param, loss);
				}
			}		
			tc++;
		}
		this.weights = w;
		return w;
	}
	
	public double[] threshold(Problem prob, Parameter param, int n_fold, int iteration) throws IOException {
		int[][] eachLabelindex = CrossValidation.getEachLabel(prob.y);
		
		int[][] trainValidIndex;	
		int i, j;
		
		double[] t = new double[this.labels.length];
		
		for(i = 0; i < n_fold; i++) {
			trainValidIndex = CrossValidation.getTrainValidIndex(eachLabelindex, n_fold, i);
			
			Problem subTrain = new Problem();
			subTrain.bias = prob.bias;
			subTrain.l = trainValidIndex[0].length;
			subTrain.n = prob.n;
			subTrain.x = new DataPoint[subTrain.l][];
			subTrain.y = new int[subTrain.l][];
			
			for(j = 0; j < subTrain.l; j++) {
				subTrain.x[j] = prob.x[trainValidIndex[0][j]];
				subTrain.y[j] = prob.y[trainValidIndex[0][j]];
			}
			
			Problem subvalid = new Problem();
			subvalid.bias = prob.bias;
			subvalid.l = trainValidIndex[1].length;
			subvalid.n = prob.n;
			subvalid.x = new DataPoint[subvalid.l][];
			subvalid.y = new int[subvalid.l][];
			
			for(j = 0; j < subvalid.l; j++) {
				subvalid.x[j] = prob.x[trainValidIndex[1][j]];
				subvalid.y[j] = prob.y[trainValidIndex[1][j]];
			}
			
			DataPoint[][] w = train(subTrain, param, iteration);
			double[][] pre = predictValues(w, subvalid.x);
			double[] scut = Scutfbr.getThreshold(this.labels, pre, subvalid.y);
			
			for(j = 0; j < t.length; j++) {
				t[j] += scut[j];
			}
		}
		
		
		DataPoint[][] w = train(prob, param, iteration);
		
		this.weights = w;
		for(j = 0; j < t.length; j++) {
			t[j] /= n_fold;
		}
		
		this.thresholds = t;
		
		return t;
	}
	
	/**
	 * 带有threshold的预测 
	 */
	public int[][] predictWithThreshold(DataPoint[][] xs) {
		int[] labels = this.labels;
		double[][] w = new double[labels.length][];
		for(int i = 0; i < w.length; i++) {
			w[i] = SparseVector.sparseVectorToArray(this.weights[labels[i]], this.prob.n);
		}
		
		int[][] predict = new int[xs.length][];
		double[] pv = new double[labels.length];
		DataPoint[] x = null;
		int counter = 0;
		for(int i = 0; i < xs.length; i++) {
			x = xs[i];
			counter = 0;
			
			for(int j = 0; j < w.length; j++) {
				pv[j] = SparseVector.innerProduct(w[j], x);
				if(pv[j] > this.thresholds[j]) {
					counter++;
				}
			}
			
			predict[i] = new int[counter];
			counter = 0;
			for(int j = 0; j < pv.length; j++) {
				if(pv[j] > this.thresholds[j]) {
					predict[i][counter++] = labels[j];
				}
			}
		}
		return predict;
	}

	/**
	 *	交叉验证过程中加入threshold, double[0] micro-f1, double[1] macro-f1, double[2] iteration 
	 * @throws IOException 
	 */
	public double[] trainValidationWithThreshold(Problem prob, Parameter param, int n_fold) throws IOException {
		int[][] eachLabelindex = CrossValidation.getEachLabel(prob.y);
		
		int[][] trainValidIndex;	
		int i, j;
		
		double[] returnValue = {0, 0, 0};
		for(i = 0; i < n_fold; i++) {
			trainValidIndex = CrossValidation.getTrainValidIndex(eachLabelindex, n_fold, i);
			
			Problem subTrain = new Problem();
			subTrain.bias = prob.bias;
			subTrain.l = trainValidIndex[0].length;
			subTrain.n = prob.n;
			subTrain.x = new DataPoint[subTrain.l][];
			subTrain.y = new int[subTrain.l][];
			
			for(j = 0; j < subTrain.l; j++) {
				subTrain.x[j] = prob.x[trainValidIndex[0][j]];
				subTrain.y[j] = prob.y[trainValidIndex[0][j]];
			}
			
			Problem subvalid = new Problem();
			subvalid.bias = prob.bias;
			subvalid.l = trainValidIndex[1].length;
			subvalid.n = prob.n;
			subvalid.x = new DataPoint[subvalid.l][];
			subvalid.y = new int[subvalid.l][];
			
			for(j = 0; j < subvalid.l; j++) {
				subvalid.x[j] = prob.x[trainValidIndex[1][j]];
				subvalid.y[j] = prob.y[trainValidIndex[1][j]];
			}
			
			double[] mmi = trainValidate(subTrain, subvalid, param);
			int iteration = (int) Math.round(mmi[2]);
			double[][] pre = predictValues(this.getWeights(), subvalid.x);
			double[] scut = Scutfbr.getThreshold(this.labels, pre, subvalid.y);
			this.thresholds = scut;
			
			int[][] pl = predictWithThreshold(subvalid.x);
			double micro_f1 = Measures.microf1(this.labels, subvalid.y, pl);
			double macro_f1 = Measures.macrof1(this.labels, subvalid.y, pl);
			
			returnValue[0] += micro_f1;
			returnValue[1] += macro_f1;
			returnValue[2] += iteration;
			

		}
		for(i = 0; i < returnValue.length; i++) {
			returnValue[i] /= n_fold;
		}
		
		return returnValue;
	}
	
	public int[] getLabels() {
		return labels;
	}

	public void setLabels(int[] labels) {
		this.labels = labels;
	}

	/**
	 * 根据目录结构返回cost
	 */
	public double[] getCost(int[][] y, int id) {
		double maxCost = this.structure.getMaxDistance();
		double pr = positiveRatio(y, id);
		
		double[] cost = new double[y.length];
		for(int i = 0; i < y.length; i++) {
			cost[i] = getCost(y[i], id, maxCost);
//			cost[i] = getCost(y[i], id, maxCost, pr);
		}
		return cost;
	}
	
	/**
	 * 
	 */
	public double getCost(int[] y, int id, double maxCost) {
		double distance = 1;
		if(Contain.contain(y, id)) {
			distance = 1;
		} else {
			double max = Double.NEGATIVE_INFINITY;
			for(int i = 0; i < y.length; i++) {
				double tempcost = this.structure.getDistance(id, y[i]);
//				tempcost = maxCost - tempcost;
				if(tempcost > max) {
					max = tempcost;
				}
			}
			distance = max;
//			distance = 3;
		}
		return Math.max(1, distance);
//		return distance;
	}
	
	
	/**
	 * 
	 */
	public double getCost(int[] y, int id, double maxCost, double pr) {
		double distance = 1;
		if(Contain.contain(y, id)) {
			distance = maxCost / pr;
		} else {
			double max = Double.NEGATIVE_INFINITY;
			for(int i = 0; i < y.length; i++) {
				double tempcost = this.structure.getDistance(id, y[i]);
				tempcost = maxCost - tempcost;
				if(tempcost > max) {
					max = tempcost;
				}
			}
			distance = max * pr;
		}
		return Math.max(distance, 1);
	}
	
	/**
	 * 正例所占比例
	 */
	public double positiveRatio(int[][] y, int id) {
		double totle = y.length;
		double counter = 0;
		for(int i = 0; i < y.length; i++) {
			if(Contain.contain(y[i], id)) {
				counter++;
			}
		}
		return counter / totle;
	}
}
