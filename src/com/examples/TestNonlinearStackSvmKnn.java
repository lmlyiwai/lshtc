package com.examples;

import static org.junit.Assert.*;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.StringTokenizer;
import java.util.Vector;

import org.junit.Test;

import com.rssvm.Measures;

public class TestNonlinearStackSvmKnn {

	@Test
	public void test() throws IOException {
		String trainfile = "E:\\hlproject\\ProcessMNIST\\train.txt";
		String testfile = "E:\\hlproject\\ProcessMNIST\\test.txt";
		
		int[] k = {1, 3, 5, 7, 11, 15, 21, 25, 31, 35, 41, 45, 51, 55, 61, 65, 71, 75, 81, 85, 91, 95};
		double[] gamma = new double[15];
		double[] c = {0.1, 1, 10, 100, 1000};
		for(int i = 0; i < 15; i++) {
			gamma[i] = Math.pow(2, i - 8);
		}
		
		svm_parameter param = setParameter(2, 3, 0.03125, 8);
		svm_problem prob = read_problem(trainfile, param);
		prob = getFirstNSamples(prob, 2000);
		
		
		TestNonlinearSvmKnnMnist tskm = new TestNonlinearSvmKnnMnist();
		
		for(int i = 0; i < gamma.length; i++) {
			for(int j = 0; j < c.length; j++) {
				tskm.crossValidation(prob, param, 5, c[j], gamma[i]);
			}
		}
		
//		for(int i = 0; i < 15; i++) {
//			for(int j = 0; j < 15; j++) {
//				double cgamma = gamma[j];
//				double cc = c[i];
//				tskm.gridSerach(prob, param, 5, cc, cgamma, k);
//			}
//		}
	
//		int[] labels = TestNonlinearSvmKnnMnist.getUniqueLabels(prob.y);
	
//		svm_problem testprob = read_problem(testfile, param);		
//		testprob = getFirstNSamples(testprob, 2000);
		

//		svm_model[] models = tskm.train(prob, param, labels);
//		double[][] tpv = tskm.predict_values(prob, models);
//		tskm.scale(tpv);
//		
//		double[][] testpv = tskm.predict_values(testprob, models);
//		tskm.scale(testpv);
//		
//		int[][] y = tskm.doubleArrayToIntMat(prob.y);
//		
//		int[][] pre = tskm.predictNear(tpv, testpv, y, 7);
//		
//		int[][] testy = tskm.doubleArrayToIntMat(testprob.y);
//		double zeroneloss = Measures.zeroOneLoss(testy, pre);
//		System.out.println("gamma = " + param.gamma + ", c = " + param.C +
//				", 0/1 loss = " + zeroneloss);
		
	}

	public static svm_parameter setParameter(int kernel_type, int degree, double gamma, double C) {
		svm_parameter param = new svm_parameter();
		// default values
		param.svm_type = svm_parameter.C_SVC;
		param.kernel_type = kernel_type;
		param.degree = degree;
		param.gamma = gamma;	// 1/num_features
		param.coef0 = 0;
		param.nu = 0.5;
		param.cache_size = 1000;
		param.C = C;
		param.eps = 1e-3;
		param.p = 0.1;
		param.shrinking = 1;
		param.probability = 0;
		param.nr_weight = 0;
		param.weight_label = new int[0];
		param.weight = new double[0];
		
		return param;
	}
	
	/**
	 * 
	 */
	public static svm_problem read_problem(String input_file_name, svm_parameter param) throws IOException
	{
		
		svm_problem prob;
		
		BufferedReader fp = new BufferedReader(new FileReader(input_file_name));
		Vector<Double> vy = new Vector<Double>();
		Vector<svm_node[]> vx = new Vector<svm_node[]>();
		int max_index = 0;

		while(true)
		{
			String line = fp.readLine();
			if(line == null) break;

			StringTokenizer st = new StringTokenizer(line," \t\n\r\f:");

			vy.addElement(atof(st.nextToken()));
			int m = st.countTokens()/2;
			svm_node[] x = new svm_node[m];
			for(int j=0;j<m;j++)
			{
				x[j] = new svm_node();
				x[j].index = atoi(st.nextToken());
				x[j].value = atof(st.nextToken());
			}
			if(m>0) max_index = Math.max(max_index, x[m-1].index);
			vx.addElement(x);
		}

		prob = new svm_problem();
		prob.l = vy.size();
		prob.x = new svm_node[prob.l][];
		for(int i=0;i<prob.l;i++)
			prob.x[i] = vx.elementAt(i);
		prob.y = new double[prob.l];
		for(int i=0;i<prob.l;i++)
			prob.y[i] = vy.elementAt(i);

		if(param.gamma == 0 && max_index > 0)
			param.gamma = 1.0/max_index;

		if(param.kernel_type == svm_parameter.PRECOMPUTED)
			for(int i=0;i<prob.l;i++)
			{
				if (prob.x[i][0].index != 0)
				{
					System.err.print("Wrong kernel matrix: first column must be 0:sample_serial_number\n");
					System.exit(1);
				}
				if ((int)prob.x[i][0].value <= 0 || (int)prob.x[i][0].value > max_index)
				{
					System.err.print("Wrong input format: sample_serial_number out of range\n");
					System.exit(1);
				}
			}

		fp.close();
		return prob;
	}
	
	private static double atof(String s)
	{
		double d = Double.valueOf(s).doubleValue();
		if (Double.isNaN(d) || Double.isInfinite(d))
		{
			System.err.print("NaN or Infinity in input\n");
			System.exit(1);
		}
		return(d);
	}

	private static int atoi(String s)
	{
		return Integer.parseInt(s);
	}
	
	/**
	 * 
	 */
	public static svm_problem getFirstNSamples(svm_problem prob, int n) {
		if(n >= prob.l) {
			return prob;
		} 
		
		svm_problem result = new svm_problem();
		result.l = n;
		result.x = new svm_node[n][];
		result.y = new double[n];		
		for(int i = 0; i < n; i++) {
			result.x[i] = prob.x[i];
			result.y[i] = prob.y[i];
		}
		return result;
	}
	
	
}
