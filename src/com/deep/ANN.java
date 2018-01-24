package com.deep;

import java.util.ArrayList;
import java.util.List;

import com.fileInputOutput.Parameter;
import com.fileInputOutput.Problem;
import com.rssvm.Measures;
import com.sparseVector.DataPoint;
import com.sparseVector.SparseVector;
import com.tools.RandomSequence;

public class ANN {
	private Problem			prob;
	private double[][][]  	weight;
	private int 			numOfClass;
	private int 			hiddenLayers;
	private int[] 			labels;
	public ANN(Problem prob, int hiddenLayers) {
		this.prob = prob;
		this.hiddenLayers = hiddenLayers;
		this.numOfClass = Matrix.uniqueLabels(prob.y);
		this.labels = new int[this.numOfClass];
		for(int i = 0; i < this.labels.length; i++) {
			this.labels[i] = i;
		}
	}
	
	/**
	 * 初始化网络权值 
	 */
	public void initWeight(Problem prob, double lr, double precision, double epoch) {
		this.weight = new double[this.hiddenLayers + 1][][];
		this.weight[0] = new double[prob.n][this.numOfClass];
		for(int i = 1; i< this.weight.length; i++) {
			this.weight[i] = new double[this.numOfClass + 1][this.numOfClass];
		}
		
		double[][] target = Matrix.fullMatrix(prob.y, this.numOfClass);
		double[][] Y = Matrix.reverseMat(target);
		
		System.out.println("初始化第" + 0 + "层");
		this.weight[0] = LeastSquare.solve(prob.x, Y, prob.n, lr, precision, epoch);
		double[][] tx = Matrix.multi(prob.x, this.weight[0]);
		double[][] X = Matrix.sigmoidMat(tx);
		X = Matrix.extendMat(X, 1);
		for(int i = 1; i < this.weight.length; i++) {
			System.out.println("初始化第" + i + "层");
			this.weight[i] = LeastSquare.solve(X, Y, lr, precision, epoch);
			tx = Matrix.multi(X, this.weight[i]);
			X = Matrix.sigmoidMat(tx);
			X = Matrix.extendMat(X, 1);
		}
	}
	
	/**
	 * 
	 */
	public void train(Problem prob, double lr, double precision, double epoch) {
//		initWeight(prob, lr, precision, 1000);
		newInitWeight(prob, lr, precision, 1000);
//		initWeight(prob);
		
		double[][] target = Matrix.fullMatrix(prob.y, this.numOfClass);
		int tc = 0;
		
		double[][][] lastDelta = new double[this.weight.length][][];
		for(int i = 0; i < this.weight.length; i++) {
			int row = this.weight[i].length;
			int col = this.weight[i][0].length;
			
			lastDelta[i] = new double[row][col];
		}
		
		while(tc++ < epoch) {
			int[] index = RandomSequence.randomSequence(prob.l);
			
			for(int i = 0; i < index.length; i++) {
				DataPoint[] x = prob.x[index[i]];
				double[] d = target[index[i]];
				double[][] output = predictOutput(x, prob.n);
				
				double[] o = output[output.length - 1];
				double[] loss = Matrix.vectorSub(d, o);
				double[] miniso = Matrix.sub(1, o);
				double[] kc = Matrix.outProduct(loss, Matrix.outProduct(o, miniso));
				kc = Matrix.scaleVec(kc, 2);
				
				double[][][] delta = new double[this.weight.length][][];
				for(int j = this.weight.length - 1; j >= 0; j--) {
					double[] ol_1 = output[j];
					if(j == 0 ) {
						//
					} else {
						ol_1 = Matrix.extendVec(ol_1, 1);
					}
					double[][] delta_j = Matrix.multi(ol_1, kc);
					delta_j = Matrix.scale(delta_j, lr);
					delta[j] = delta_j;
					
					double[] cutOl_1 = Matrix.cutVec(ol_1);
					double[] newkc = new double[this.weight[j].length - 1];
					for(int k = 0; k < newkc.length; k++) {
						newkc[k] = Matrix.innerProcuct(kc, this.weight[j][k]);
					}
					
					double[] minisOl_1 = Matrix.sub(1, cutOl_1);
					kc = Matrix.outProduct(newkc, Matrix.outProduct(cutOl_1, minisOl_1));
				}
				
				for(int j = 0; j < this.weight.length; j++) {
					lastDelta[j] = Matrix.scale(lastDelta[j], 0.9);
					delta[j] = Matrix.matrixAdd(delta[j], lastDelta[j]);
					lastDelta[j] = delta[j];
				}
				
				for(int j = 0; j < this.weight.length; j++) {
					this.weight[j] = Matrix.matrixAdd(this.weight[j], delta[j]);
				}
			}
			
			double obj = 0;
			int[][] pl = new int[prob.l][];
			for(int i = 0; i < prob.l; i++) {
				double[] pv = predictFinalOutput(prob.x[i], prob.n);
				double[] sub = Matrix.vectorSub(pv, target[i]);
				double inp = Matrix.innerProcuct(sub, sub);
				obj = obj + inp;
				
				pl[i] = predict(prob.x[i], 0.5, prob.n);
			}
			double microf1 = Measures.microf1(this.labels, prob.y, pl);
			System.out.println("obj = " + obj + ", Micro-F1 = " + microf1);
		}
	}
	
	/**
	 * 预测输出，返回每一层输出值
	 */
	public double[][] predictOutput(DataPoint[] x, int dimx) {
		double[][] output = new double[this.hiddenLayers+2][];
		output[0] = SparseVector.sparseVectorToArray(x, dimx);
		output[1] = Matrix.multi(x, this.weight[0]);
		output[1] = Matrix.sigmoidVec(output[1]);                  //
		for(int i = 2; i < output.length; i++) {
			double[] tx = Matrix.extendVec(output[i-1], 1);
			output[i] = Matrix.multi(tx, this.weight[i-1]);
			output[i] = Matrix.sigmoidVec(output[i]);
		}
		return output;
	}
	
	/**
	 * 
	 */
	public int[] predict(DataPoint[] x, double threshold, int dimx) {
		double[][] output = predictOutput(x, dimx);
		double[] finalOutput = output[output.length - 1];
		List<Integer> list = new ArrayList<Integer>();
		for(int i = 0; i < finalOutput.length; i++) {
			if(finalOutput[i] > threshold) {
				list.add(i);
			}
		}
		int[] pl = new int[list.size()];
		for(int i = 0; i < pl.length; i++) {
			pl[i] = list.get(i);
		}
		return pl;
	}
	
	/**
	 * 网络输出值
	 */
	public double[] predictFinalOutput(DataPoint[] x, int dimx) {
		double[][] out = predictOutput(x, dimx);
		return out[out.length - 1];
	}

	public int[] getLabels() {
		return labels;
	}

	public void setLabels(int[] labels) {
		this.labels = labels;
	} 
	
	/**
	 * 初始化网络权值 
	 */
	public void initWeight(Problem prob) {
		this.weight = new double[this.hiddenLayers + 1][][];
		this.weight[0] = new double[prob.n][this.numOfClass];
		Matrix.randInitMat(this.weight[0], -0.5, 0.5);
		for(int i = 1; i< this.weight.length; i++) {
			this.weight[i] = new double[this.numOfClass + 1][this.numOfClass];
			Matrix.randInitMat(this.weight[i], -0.5, 0.5);
		}
	}
	
	/**
	 * 
	 */
	public double train(Problem prob, Problem test, double lr, double precision, double epoch) {
//		initWeight(prob, lr, precision, 1000);
		newInitWeight(prob, lr, precision, 1000);
		
		double[][] target = Matrix.fullMatrix(prob.y, this.numOfClass);
		int tc = 0;
		
		double[][][] lastDelta = new double[this.weight.length][][];
		for(int i = 0; i < this.weight.length; i++) {
			int row = this.weight[i].length;
			int col = this.weight[i][0].length;
			
			lastDelta[i] = new double[row][col];
		}
		
		double lastPerf = Double.NEGATIVE_INFINITY;
		while(tc++ < epoch) {
			int[] index = RandomSequence.randomSequence(prob.l);
			
			for(int i = 0; i < index.length; i++) {
				DataPoint[] x = prob.x[index[i]];
				double[] d = target[index[i]];
				double[][] output = predictOutput(x, prob.n);
				
				double[] o = output[output.length - 1];
				double[] loss = Matrix.vectorSub(d, o);
				double[] miniso = Matrix.sub(1, o);
				double[] kc = Matrix.outProduct(loss, Matrix.outProduct(o, miniso));
				kc = Matrix.scaleVec(kc, 2);
				
				double[][][] delta = new double[this.weight.length][][];
				for(int j = this.weight.length - 1; j >= 0; j--) {
					double[] ol_1 = output[j];
					if(j == 0 ) {
						//
					} else {
						ol_1 = Matrix.extendVec(ol_1, 1);
					}
					double[][] delta_j = Matrix.multi(ol_1, kc);
					delta_j = Matrix.scale(delta_j, lr);
					delta[j] = delta_j;
					
					double[] cutOl_1 = Matrix.cutVec(ol_1);
					double[] newkc = new double[this.weight[j].length - 1];
					for(int k = 0; k < newkc.length; k++) {
						newkc[k] = Matrix.innerProcuct(kc, this.weight[j][k]);
					}
					
					double[] minisOl_1 = Matrix.sub(1, cutOl_1);
					kc = Matrix.outProduct(newkc, Matrix.outProduct(cutOl_1, minisOl_1));
				}
				
				for(int j = 0; j < this.weight.length; j++) {
					lastDelta[j] = Matrix.scale(lastDelta[j], 0.9);
					delta[j] = Matrix.matrixAdd(delta[j], lastDelta[j]);
					lastDelta[j] = delta[j];
				}
				
				for(int j = 0; j < this.weight.length; j++) {
					this.weight[j] = Matrix.matrixAdd(this.weight[j], delta[j]);
				}
			}
			
			double obj = 0;
			int[][] pl = new int[prob.l][];
			for(int i = 0; i < prob.l; i++) {
				double[] pv = predictFinalOutput(prob.x[i], prob.n);
				double[] sub = Matrix.vectorSub(pv, target[i]);
				double inp = Matrix.innerProcuct(sub, sub);
				obj = obj + inp;
				
				pl[i] = predict(prob.x[i], 0.5, prob.n);
			}
			
			int[][] testpl = new int[test.l][];
			for(int i = 0; i < testpl.length; i++) {
				testpl[i] = predict(test.x[i], 0.5, test.n);
			}
			
			double microf1 = Measures.microf1(this.labels, prob.y, pl);
			double macrof1 = Measures.macrof1(this.labels, prob.y, pl);
			
			double testmif1 = Measures.microf1(this.labels, test.y, testpl);
			double testmaf1 = Measures.macrof1(this.labels, test.y, testpl);
			
			if(testmaf1 < lastPerf) {
//				break;
			}
			lastPerf = testmaf1;
			System.out.printf("obj = %.4f, MiF1 = %.4f, MaF1 = %.4f, Micro-F1 = %.4f, Macro-F1 = %.4f.\n",
					obj, testmif1, testmaf1, microf1, macrof1);
		}
		return tc;
	}
	
	/**
	 * 初始化网络权值 ，并计算损失之和
	 */
	public void newInitWeight(Problem prob, double lr, double precision, double epoch) {
		this.weight = new double[this.hiddenLayers + 1][][];
		this.weight[0] = new double[prob.n][this.numOfClass];
		for(int i = 1; i< this.weight.length; i++) {
			this.weight[i] = new double[this.numOfClass + 1][this.numOfClass];
		}
		
		double[][] target = Matrix.fullMatrix(prob.y, this.numOfClass);
		double[][] Y = Matrix.reverseMat(target);
		
		System.out.print("初始化第" + 0 + "层");
		this.weight[0] = LeastSquare.newSolve(prob.x, Y, prob.n, lr, precision, 10);
		double[][] tx = Matrix.multi(prob.x, this.weight[0]);
		double[][] X = Matrix.sigmoidMat(tx);
		double los = loss(X, target);
		System.out.println(", obj = " + los);
		
		double lastLoss = los;
		X = Matrix.extendMat(X, 1);
		
		for(int i = 1; i < this.weight.length; i++) {
			System.out.print("初始化第" + i + "层");
			double[][] berfX = X;
			this.weight[i] = LeastSquare.solve(X, Y, lr, precision, epoch);
			tx = Matrix.multi(X, this.weight[i]);
			X = Matrix.sigmoidMat(tx);
			los = loss(X, target);
			X = Matrix.extendMat(X, 1);
			
			if(los > lastLoss && i > 1) {
				Matrix.copyMat(this.weight[i-1], this.weight[i]);
				X = berfX;
				tx = Matrix.multi(X, this.weight[i]);
				X = Matrix.sigmoidMat(tx);
				los = loss(X, target);	
				X = Matrix.extendMat(X, 1);
			}
			
			lastLoss = los;
			System.out.println(", obj = " + los);
		}
	}
	
	/**
	 * 
	 */
	public double sumLoss(double[][] X, double[][] target, double[][] w) {
		double[][] xw = Matrix.multi(X, w);
		xw = Matrix.sigmoidMat(xw);
		double loss = 0;
		for(int i = 0; i < target.length; i++) {
			double[] temp = Matrix.vectorSub(xw[i], target[i]);
			double inp = SparseVector.innerProduct(temp, temp);
			loss += inp;
		}
		return loss;
	}
	
	/**
	 * 
	 */
	public double sumLoss(DataPoint[][] x, double[][] target, double[][] w, int n) {
		double[][] xw = new double[x.length][];
		for(int i = 0; i < xw.length; i++) {
			xw[i] = Matrix.multi(x[i], w);
		}
		xw = Matrix.sigmoidMat(xw);
		double loss = 0;
		for(int i = 0; i < target.length; i++) {
			double[] temp = Matrix.vectorSub(xw[i], target[i]);
			double inp = SparseVector.innerProduct(temp, temp);
			loss += inp;
		}
		return loss;
	}
	
	/**
	 * 
	 */
	public double loss(double[][] y, double[][] t) {
		double loss = 0;
		for(int i = 0; i < y.length; i++) {
			double[] sub = Matrix.vectorSub(y[i], t[i]);
			double inp = Matrix.innerProcuct(sub, sub);
			loss = loss + inp;
		}
		return loss;
	}
	
	/**
	 * 
	 */
	public double crossValidation(Problem prob, double lr, double precision, double epoch, int n_fold) {
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
			
		
			
			int[][] predictLabel = new int[valid.l][];
			
			for(int j = 0; j < validIndex.length; j++) {
				pre[validIndex[j]] = predictLabel[j];
			}
		}
		double microf1 = Measures.microf1(this.labels, prob.y, pre);
		double macrof1 = Measures.macrof1(this.labels, prob.y, pre);
		double hammingloss = Measures.averageSymLoss(prob.y, pre);
		System.out.println("c = " + ", Micro-F1 = " + microf1 + ", Macro-F1 = " 
				+ macrof1 + ", Hamming Loss = " + hammingloss);
		double[] perf = {microf1, macrof1, hammingloss};
		return 0;
	}
	
	/**
	 * 
	 */
	public double newTrain(Problem prob, Problem test, double lr, double precision, double epoch) {
//		initWeight(prob, lr, precision, 1000);                 //
		newInitWeight(prob, 0.01, 0.01, 1000);           //保证输入在指定范围
		
		double[][] target = Matrix.fullMatrix(prob.y, this.numOfClass);
		int tc = 0;
		
		double[][][] lastDelta = new double[this.weight.length][][];
		for(int i = 0; i < this.weight.length; i++) {
			int row = this.weight[i].length;
			int col = this.weight[i][0].length;
			
			lastDelta[i] = new double[row][col];
		}
		
		double lastPerf = Double.NEGATIVE_INFINITY;
		while(tc++ < epoch) {
			int[] index = RandomSequence.randomSequence(prob.l);
			
			for(int i = 0; i < index.length; i++) {
				DataPoint[] x = prob.x[index[i]];
				double[] d = target[index[i]];
				double[][] output = predictOutput(x, prob.n);
				
				double[] o = output[output.length - 1];           //样本x经过网络的输出
				double[] loss = Matrix.vectorSub(d, o);
				double[] miniso = Matrix.sub(1, o);
				double[] kc = Matrix.outProduct(loss, Matrix.outProduct(o, miniso));
				kc = Matrix.scaleVec(kc, 2);
				
				double tlos = Matrix.innerProcuct(loss, loss);
				
				double[][][] delta = new double[this.weight.length][][];
				for(int j = this.weight.length - 1; j >= 0; j--) {
					double[] ol_1 = output[j];
					if(j == 0 ) {
						//
					} else {
						ol_1 = Matrix.extendVec(ol_1, 1);
					}
					double[][] delta_j = Matrix.multi(ol_1, kc);
					delta_j = Matrix.scale(delta_j, lr);
					delta[j] = delta_j;
					
					double[] cutOl_1 = Matrix.cutVec(ol_1);
					double[] newkc = new double[this.weight[j].length - 1];
					for(int k = 0; k < newkc.length; k++) {
						newkc[k] = Matrix.innerProcuct(kc, this.weight[j][k]);
					}
					
					double[] minisOl_1 = Matrix.sub(1, cutOl_1);
					kc = Matrix.outProduct(newkc, Matrix.outProduct(cutOl_1, minisOl_1));
				}
				
				for(int j = 0; j < this.weight.length; j++) {
					lastDelta[j] = Matrix.scale(lastDelta[j], 0.9);
					delta[j] = Matrix.matrixAdd(delta[j], lastDelta[j]);
					lastDelta[j] = delta[j];
				}
					
				delta = scaleDelta(x, prob.n, delta, this.weight, d, tlos);
				lastDelta = delta;
				
				for(int j = 0; j < this.weight.length; j++) {
					this.weight[j] = Matrix.matrixAdd(this.weight[j], delta[j]);
				}
			}
			
			double obj = 0;
			int[][] pl = new int[prob.l][];
			for(int i = 0; i < prob.l; i++) {
				double[] pv = predictFinalOutput(prob.x[i], prob.n);
				double[] sub = Matrix.vectorSub(pv, target[i]);
				double inp = Matrix.innerProcuct(sub, sub);
				obj = obj + inp;
				
				pl[i] = predict(prob.x[i], 0.5, prob.n);
			}
			
			int[][] testpl = new int[test.l][];
			for(int i = 0; i < testpl.length; i++) {
				testpl[i] = predict(test.x[i], 0.5, test.n);
			}
			
			double microf1 = Measures.microf1(this.labels, prob.y, pl);
			double macrof1 = Measures.macrof1(this.labels, prob.y, pl);
			
			double testmif1 = Measures.microf1(this.labels, test.y, testpl);
			double testmaf1 = Measures.macrof1(this.labels, test.y, testpl);
			
			if(testmaf1 < lastPerf) {
//				break;
			}
			lastPerf = testmaf1;
			System.out.printf("obj = %.4f, MiF1 = %.4f, MaF1 = %.4f, Micro-F1 = %.4f, Macro-F1 = %.4f, it = %d\n",
					obj, testmif1, testmaf1, microf1, macrof1, tc);
		}
		return tc;
	}
	
	/**
	 * 预测输出，返回每一层输出值
	 */
	public double[][] predictOutput(DataPoint[] x, int dimx, double[][][] delta) {
		double[][] output = new double[this.hiddenLayers+2][];
		output[0] = SparseVector.sparseVectorToArray(x, dimx);
		output[1] = Matrix.multi(x, delta[0]);
		output[1] = Matrix.sigmoidVec(output[1]);                  //
		for(int i = 2; i < output.length; i++) {
			double[] tx = Matrix.extendVec(output[i-1], 1);
			output[i] = Matrix.multi(tx, delta[i-1]);
			output[i] = Matrix.sigmoidVec(output[i]);
		}
		return output;
	}
	
	/**
	 * 网络输出值
	 */
	public double[] predictFinalOutput(DataPoint[] x, int dimx, double[][][] delta) {
		double[][] out = predictOutput(x, dimx, delta);
		return out[out.length - 1];
	}
	
	/**
	 * 
	 */
	public double[][][] scaleDelta(DataPoint[] x, int dimx, double[][][] delta, double[][][] w, double[] objvec, double obj) {
		double[] s = {1, 0.7, 0.5, 0.2, 0};
		double[][][] cd = null;
		double[][][] nw = null;
		for(int i = 0; i < s.length; i++) {
			cd = Matrix.scaleMat(delta, s[i]);
			nw = Matrix.matAdd(w, delta);
			double[] fp = predictFinalOutput(x, dimx, nw);
			fp = Matrix.vectorSub(objvec, fp);
			double tobj = Matrix.innerProcuct(fp, fp);
			if(tobj <= obj) {
				break;
			}
		}
		return cd;
	}
	
	/**
	 * 
	 */
	public void newTrain(Problem prob, double lr, double precision, double epoch) {
//		initWeight(prob, lr, precision, 1000);
		newInitWeight(prob, lr, precision, 1000);
//		initWeight(prob);
		
		double[][] target = Matrix.fullMatrix(prob.y, this.numOfClass);
		int tc = 0;
		
		double[][][] lastDelta = new double[this.weight.length][][];
		for(int i = 0; i < this.weight.length; i++) {
			int row = this.weight[i].length;
			int col = this.weight[i][0].length;
			
			lastDelta[i] = new double[row][col];
		}
		
		while(tc++ < epoch) {
			int[] index = RandomSequence.randomSequence(prob.l);
			
			for(int i = 0; i < index.length; i++) {
				DataPoint[] x = prob.x[index[i]];
				double[] d = target[index[i]];
				double[][] output = predictOutput(x, prob.n);
				
				double[] o = output[output.length - 1];
				double[] loss = Matrix.vectorSub(d, o);
				double[] miniso = Matrix.sub(1, o);
				double[] kc = Matrix.outProduct(loss, Matrix.outProduct(o, miniso));
				kc = Matrix.scaleVec(kc, 2);
				
				double tloss = Matrix.innerProcuct(loss, loss);
				
				double[][][] delta = new double[this.weight.length][][];
				for(int j = this.weight.length - 1; j >= 0; j--) {
					double[] ol_1 = output[j];
					if(j == 0 ) {
						//
					} else {
						ol_1 = Matrix.extendVec(ol_1, 1);
					}
					double[][] delta_j = Matrix.multi(ol_1, kc);
					delta_j = Matrix.scale(delta_j, lr);
					delta[j] = delta_j;
					
					double[] cutOl_1 = Matrix.cutVec(ol_1);
					double[] newkc = new double[this.weight[j].length - 1];
					for(int k = 0; k < newkc.length; k++) {
						newkc[k] = Matrix.innerProcuct(kc, this.weight[j][k]);
					}
					
					double[] minisOl_1 = Matrix.sub(1, cutOl_1);
					kc = Matrix.outProduct(newkc, Matrix.outProduct(cutOl_1, minisOl_1));
				}
				
				for(int j = 0; j < this.weight.length; j++) {
					lastDelta[j] = Matrix.scale(lastDelta[j], 0.9);
					delta[j] = Matrix.matrixAdd(delta[j], lastDelta[j]);
					lastDelta[j] = delta[j];
				}
				
				delta = scaleDelta(x, prob.n, delta, this.weight, d, tloss);
				lastDelta = delta;
				
				for(int j = 0; j < this.weight.length; j++) {
					this.weight[j] = Matrix.matrixAdd(this.weight[j], delta[j]);
				}
			}
			
			double obj = 0;
			int[][] pl = new int[prob.l][];
			for(int i = 0; i < prob.l; i++) {
				double[] pv = predictFinalOutput(prob.x[i], prob.n);
				double[] sub = Matrix.vectorSub(pv, target[i]);
				double inp = Matrix.innerProcuct(sub, sub);
				obj = obj + inp;
				
				pl[i] = predict(prob.x[i], 0.5, prob.n);
			}
			double microf1 = Measures.microf1(this.labels, prob.y, pl);
			System.out.println("obj = " + obj + ", Micro-F1 = " + microf1);
		}
	}
}
