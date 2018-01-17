package com.DeepNetworkWithSVM;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.List;

import javax.xml.crypto.Data;

import com.fileInputOutput.Parameter;
import com.fileInputOutput.Problem;
import com.sparseVector.DataPoint;
import com.sparseVector.SparseVector;

public class TrainDeepSVM {
	private int 					layers;
	private Problem 			train;
	private Parameter[] 		params;
	private int[] 					ulabels;
	private static final double 	threshold = 0.0;
	
	public TrainDeepSVM(Problem train, Parameter[] params, int layers){
		this.train = train;
		this.params = params;
		this.layers = layers;
		this.ulabels = getUlabels(train.y);
	}
	
	/*
	 * 训练支持向量机，并将所得拉格朗日因子写入文件outputFile 
	 */
	public void train(String outputFileBase) throws IOException {
		BufferedWriter out = null;
		
		for(int i = 0; i < this.ulabels.length; i++) {
			int label = this.ulabels[i];
			int[] y = binaryLabel(this.train.y, label);
//			System.out.println(label);
			String wfile = outputFileBase + "\\" + label;
			out = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(wfile)));
			DataPoint[][] x = copyMatrix(this.train.x);
			for(int j = 0; j < this.layers - 1; j++) {
				double[] alpha = Linear.train(x, y, this.params[j], getDim(x));
				writeSupportVecotrsToFile(out, alpha, x, y);
				x = transformSamples(x, alpha, y);
//				scale(x);                                                                // 1 若做归一化，三处需改动。// 2 // 3
			}
			double[] alpha = Linear.train(x, y, this.params[this.params.length-1], getDim(x));
			writeSupportVecotrsToFile(out, alpha, x, y);
			out.close();
		}
		
	}
	
	private void writeSupportVecotrsToFile(BufferedWriter out, double[] alpha, DataPoint[][] x, int[] y) throws IOException {
		if(out == null || alpha == null || x == null) {
			return;
		}
		int count = 0;
		for(int i = 0; i < alpha.length; i++) {
			if(alpha[i] > TrainDeepSVM.threshold) {
				count++;
			}
		}
		out.write(count+"\n");
		for(int  i = 0; i < alpha.length; i++) {
			if(alpha[i] > TrainDeepSVM.threshold) {
				out.write(i + " " + alpha[i] + " " + y[i] + " ");
				out.write("\n");
			}
		}
	}
	
	private int[] getUlabels(int[][] y){
		if(y == null)
		{
			return new int[0];
		}
		List<Integer> list = new ArrayList<Integer>();
		for(int i = 0; i < y.length; i++) {
			for(int j = 0; j < y[i].length; j++) {
				if(!list.contains(y[i][j])) {
					list.add(y[i][j]);
				}
			}
		}
		int[] result = new int[list.size()];
		for(int i = 0; i < result.length; i++) {
			result[i] = list.get(i);
		}
		return result;
	}
	
	private int[] binaryLabel(int[][] y, int label) {
		if(y == null) {
			return null;
		}
		int[] result = new int[y.length];
		for(int i = 0; i < result.length; i++) {
			if(contain(y[i], label)) {
				result[i] = 1;
			} else {
				result[i] = -1;
			}
		}
		return result;
	}
	
	private boolean contain(int[] array, int e) {
		if(array == null) {
			return false;
		}
		boolean flag = false;
		for(int i = 0; i < array.length; i++) {
			if(array[i] == e) {
				flag = true;
				break;
			}
		}
		return flag;
	}
	
	private DataPoint[][] copyMatrix(DataPoint[][] x) {
		if(x == null) {
			return null;
		}
		DataPoint[][] result = new DataPoint[x.length][];
		for(int i = 0; i < result.length; i++) {
			result[i] = new DataPoint[x[i].length];
			for(int j = 0; j < result[i].length; j++) {
				result[i][j] = new DataPoint(x[i][j].index, x[i][j].value);
			}
		}
		return result;
	}
	
	/**
	 * 训练样本与支持向量做内积 
	 */
	private DataPoint[][] transformSamples(DataPoint[][] x, double[] alpha, int[] y) {
		if(x == null || alpha == null || y == null) {
			return null;
		}
		List<Integer> list = new ArrayList<Integer>();
		for(int i = 0; i < alpha.length; i++) {
			if(alpha[i] > TrainDeepSVM.threshold) {
				list.add(i);
			}
		}
		int col = list.size();
		DataPoint[][] result = new DataPoint[x.length][];
		for(int i = 0; i < result.length; i++) {
			result[i] = new DataPoint[col];
			for(int j = 0; j < col; j++) {
				int index = j + 1;
				double value = alpha[list.get(j)] * y[list.get(j)] * SparseVector.innerProduct(x[i], x[list.get(j)]);
				result[i][j] = new DataPoint(index, value);
			} 
		}
		return result;
	}
	
	/**
	 * 将样本模长归一化为1 
	 */
	private void scale(DataPoint[][] x) {
		if(x == null) {
			return;
		}
		for(int i = 0; i < x.length; i++) {
			double norm = 0.0;
			for(DataPoint dp : x[i]) {
				norm += dp.value * dp.value;
			}
			norm = Math.sqrt(norm);
			for(DataPoint dp : x[i]) {
				dp.value = dp.value / norm;
			}
		}
	}
	
	public void extendMatrixWithBias(DataPoint[][] x, double bias) {
		if(x == null) {
			return;
		}
		int maxIndex = -1;
		for(int i = 0; i < x.length; i++) {
			for(int j = 0; j < x[i].length; j++) {
				if(x[i][j].index > maxIndex) {
					maxIndex = x[i][j].index;
				}
			}
		}
		maxIndex++;
		for(int i = 0; i < x.length; i++) {
			DataPoint[] temp = new DataPoint[x[i].length + 1];
			for(int j = 0; j < temp.length - 1; j++) {
				temp[j] = x[i][j];
			}
			temp[temp.length-1] = new DataPoint(maxIndex, bias);
		}
	}
	
	private int getDim(DataPoint[][] x) {
		if(x == null) {
			return -1;
		}
		int maxDim = -1;
		for(int i = 0; i < x.length; i++) {
			for(int j = 0; j < x[i].length; j++) {
				if(x[i][j].index > maxDim) {
					maxDim = x[i][j].index;
				}
			}
		}
		return maxDim;
	}
	
	/**
	 *@param x测试样本集, classPath分类器所在目录
	 *@return 返回值为二维数组，一行代表一个样本在各层的预测类标 
	 */
	public int[][] predict(DataPoint[][] x, String classPath) throws IOException {
		if(x == null || classPath == null) {
			return null;
		}
		
		double[][] maxPredictOfEachLayer = new double[x.length][this.layers];
		int[][] predictLabelsOfEachLayer = new int[x.length][this.layers];
		for (int i = 0; i < maxPredictOfEachLayer.length; i++) {
			for (int j = 0; j < maxPredictOfEachLayer[i].length; j++) {
				maxPredictOfEachLayer[i][j] = Double.NEGATIVE_INFINITY;
				predictLabelsOfEachLayer[i][j] = -1;
			}
		}
		
		File baseFile = new File(classPath);
		String[] labels = baseFile.list();
		for(int i = 0; i < labels.length; i++) {
			int label = Integer.parseInt(labels[i]);
			String weightFile = classPath + "\\" + labels[i];
			
			double[][][] indexOfSupportVectors = readSV(weightFile);
			DataPoint[][][] supportVectorOfEachLayer = getSVs(this.train.x, indexOfSupportVectors);
			double[][] predictValuesOfEachLayer = getOutputOfEachLayer(x, supportVectorOfEachLayer);
			
			for (int row = 0; row < maxPredictOfEachLayer.length; row++) {
				for (int col = 0; col < maxPredictOfEachLayer[row].length; col++) {
					if(predictValuesOfEachLayer[row][col] > maxPredictOfEachLayer[row][col]) {
						maxPredictOfEachLayer[row][col] = predictValuesOfEachLayer[row][col];
						predictLabelsOfEachLayer[row][col] = label;
					}
				}
			}
		}
		return predictLabelsOfEachLayer;
	}
	
	/**
	 *  计算输入样本进过支持向量网络各层的输出值
	 */
	private double[][] getOutputOfEachLayer(DataPoint[][] x, DataPoint[][][] sv) {
		if (x == null || sv == null) {
			return null;
		}
		
		double[][] predictValues = new double[x.length][this.layers];
		for (int i = 0; i < predictValues.length; i++) {
			DataPoint[] xi = x[i];
			predictValues[i] = getSingleSampleOutput(xi, sv);
		}
		return predictValues;
	}
	
	/**
	 * 
	 */
	private double[] getSingleSampleOutput(DataPoint[] x, DataPoint[][][] sv) {
		if(x == null || sv == null) {
			return null;
		}
		
		DataPoint[] copyx = copySparseVector(x);
		double[] predictValueOfEachLayer = new double[sv.length];
		for (int i = 0; i < predictValueOfEachLayer.length; i++) {
			DataPoint[][] svi = sv[i];
			DataPoint[] nextLayerx = new DataPoint[svi.length];
			double sumOfPredict = 0.0;
			for (int j = 0; j < svi.length; j++) {
				int index = j + 1;
				double value = SparseVector.innerProduct(copyx, svi[j]);
				sumOfPredict += value;
				nextLayerx[j] = new DataPoint(index, value);
			}
//			normalizeVector(nextLayerx);                                         // 3
			copyx = nextLayerx;
			predictValueOfEachLayer[i] = sumOfPredict;
		}
		
		
		return predictValueOfEachLayer;
	}
	
	/**
	 * @param x要复制的对象
	 * @return x的拷贝，x本身不变化
	 */
	private DataPoint[] copySparseVector(DataPoint[] x) {
		if (x == null) {
			return null;
		}
		DataPoint[] copyOfX = new DataPoint[x.length];
		for (int i = 0; i < copyOfX.length; i++) {
			copyOfX[i] = new DataPoint(x[i].index, x[i].value);
		}
		return copyOfX;
	}
	
	/**
	 * 获得每层的支持向量与拉格朗日系数及类别的乘积 
	 */
	private DataPoint[][][] getSVs(DataPoint[][] oriSamples, double[][][] svs) {
		if (oriSamples == null || svs == null) {
			return null;
		} 
		DataPoint[][][] result = new DataPoint[svs.length][][];
		double[][] svs0 = svs[0];
		DataPoint[][] result0 = new DataPoint[svs0.length][];
		double scale = 0;
		for (int i = 0; i < result0.length; i++) {
			scale = svs0[i][1] * svs0[i][2];
			result0[i] = scaleSparseVector(oriSamples[(int)svs0[i][0]], scale);
		}
		result[0] = result0;

		for (int i = 1; i < result.length; i++) {
			double[][] svsi = svs[i];
			result[i] = copySparseVector(oriSamples, svsi);
			for (int j = 0; j < i; j++) {
				DataPoint[][] preSVs = result[j];
				for (int m = 0; m < result[i].length; m++) {
					DataPoint[] retm = result[i][m];                            //
					result[i][m] = new DataPoint[preSVs.length];
					for (int n = 0; n < preSVs.length; n++) {
						int index = n + 1;
						double value = SparseVector.innerProduct(retm, preSVs[n]);
						result[i][m][n] = new DataPoint(index, value);
					}
//					normalizeVector(result[i][m]);                        // 2
					scale = svsi[m][1] * svsi[m][2];
					for (DataPoint dp : result[i][m]) {
						dp.value = dp.value * scale;
					}
				}
			}
//SP.showMatrix(result[i]);
		}
		return result;
	}
	
	/**
	 * 将vec模长归一化为1
	 */
	private void normalizeVector(DataPoint[] vec) {
		if (vec == null) {
			return;
		}
		
		double sum = 0.0;
		for (DataPoint dp : vec) {
			sum += dp.value * dp.value;
		}
		sum = Math.pow(sum, 0.5);
		
		for (DataPoint dp : vec) {
			dp.value = dp.value / sum;
		}
	}
	
	/**
	 * 
	 */
	private DataPoint[][] copySparseVector(DataPoint[][] orisp, double[][] sv) {
		if(orisp == null || sv == null) {
			return null;
		}
		DataPoint[][] result = new DataPoint[sv.length][];
		for (int i = 0; i < result.length; i++) {
			result[i] = new DataPoint[orisp[(int)sv[i][0]].length];
			for (int j = 0; j < result[i].length; j++) {
				DataPoint dp = orisp[(int)sv[i][0]][j];
				result[i][j] = new DataPoint(dp.index, dp.value);
			}
		}
		return result;
	}
	
	/**
	 * dp.value * scale，dp本身不改变，返回新的对象
	 */
	private DataPoint[] scaleSparseVector(DataPoint[] dp, double scale) {
		if (dp == null) {
			return null;
		} 
		DataPoint[] result = new DataPoint[dp.length];
		for (int i = 0; i < result.length; i++) {
			result[i] = new DataPoint(dp[i].index, dp[i].value * scale);
		}
		return result;
	}
	
	/**
	 * 读取每层支持向量对应编号，拉格朗日系数，类标 
	 * 
	 */
	private double[][][] readSV(String filename) {
		BufferedReader in = null;
		double[][][] result = null;
		
		try {
			in = new BufferedReader(new InputStreamReader(
					new FileInputStream(filename)));
			result = new double[this.layers][][];
			String line = null;
			int numOfSV = 0;
			int index = 0;
			while ((line = in.readLine()) != null) {
				numOfSV = Integer.parseInt(line);
				double[][] sv = new double[numOfSV][3];
				String[] splits = null;
				for (int i = 0; i < numOfSV; i++) {
					line = in.readLine();
					splits = line.split("\\s+");
					for (int j = 0; j < splits.length; j++) {
						sv[i][j] = Double.parseDouble(splits[j]);
					}
				}
				result[index++] = sv;
			}
			in.close();
			return result;
		} catch(IOException e) {
			e.printStackTrace();
			return null;
		}
	}
	
	/**
	 * 给定类标，获得各层输出
	 * x 为输入
	 */
	public double[][] predictSingleLabel(DataPoint[][] x, double[][] alphas, int[] y) {
		if(x == null || alphas == null || y == null) {
			return null;
		}
		double[][] result = new double[this.layers][x.length];
		DataPoint[][] cx = copyMatrix(this.train.x);
		for(int i = 0; i < this.layers; i++) {
			double[][] tx = transform(cx, alphas[i], y, x);
			for(int j = 0; j < result.length; j++) {
				result[i][j] = sumVector(tx[j]);
			}
			cx = transformSamples(cx, alphas[i], y);
		}
		return result;
	}
	
	/**
	 * 测试样本转换
	 */
	private double[][] transform(DataPoint[][] x, double[] alpha, int[] y, DataPoint[][] tx) {
		if(x == null || alpha == null || tx == null) {
			return null;
		}
		int[] indexs = getThresholdIndex(alpha);
		double[][] result = new double[tx.length][indexs.length];
		for(int i = 0; i < result.length; i++) {
			for(int j = 0; j < indexs.length; j++) {
				result[i][j] = alpha[indexs[j]] * y[indexs[j]] * SparseVector.innerProduct(tx[i], x[indexs[j]]); 
			}
		}
		return result;
	}
	
	private double sumVector(double[] vec) {
		if(vec == null) {
			return Double.NEGATIVE_INFINITY;
		}
		double result = 0.0;
		for(int i = 0; i < vec.length; i++) {
			result += vec[i];
		}
		return result;
	}
	
	private int[] getThresholdIndex(double[] alpha) {
		List<Integer> list = new ArrayList<Integer>();
		for(int i = 0; i < alpha.length; i++) {
			if(alpha[i] > TrainDeepSVM.threshold) {
				list.add(i);
			}
		}
		int[] result = new int[list.size()];
		for(int i = 0; i < result.length; i++) {
			result[i] = list.get(i);
		}
		return result;
	}

	public int[] getUlabels() {
		return ulabels;
	}

	public void setUlabels(int[] ulabels) {
		this.ulabels = ulabels;
	}
	
	
}
