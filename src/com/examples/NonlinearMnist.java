package com.examples;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.StringTokenizer;
import java.util.Vector;
public class NonlinearMnist {
	public static void main(String[] args) throws IOException {
//		String trainfile = "F:\\java\\ProcessMNIST\\smallTrain.txt";
//		String testfile = "F:\\java\\ProcessMNIST\\smallTest.txt";
		
//		String trainfile = "F:\\java\\ProcessMNIST\\train.txt";
//		String testfile = "F:\\java\\ProcessMNIST\\test.txt";
		
		String trainfile = "train.txt";
		String testfile = "train.txt";
		
		svm_parameter param = setParameter(0, 3, 0.001, 1);
		svm_problem prob = read_problem(trainfile, param);
		prob = getFirstNSamples(prob, 2000);
		
		
		svm_problem testprob = read_problem(testfile, param);
		testprob = getFirstNSamples(testprob, 1000);
		
		double[] gamma = new double[15];
		double[] c = new double[15];
		for(int i = 0; i < gamma.length; i++) {
			gamma[i] = Math.pow(2, i - 7);
			c[i] = Math.pow(2, i - 7);
		}
		
		for(int m = 0; m < gamma.length; m++) {
			for(int n = 0; n < c.length; n++) {
				
				param = setParameter(0, 3, gamma[m], c[n]);
				System.out.println("gamma = " + gamma[m] + ", c = " + c[n]);
				svm.svm_set_print_string_function(null);
				svm_model model = svm.svm_train(prob, param);
				
				double[] pre = new double[testprob.x.length];
				for(int i = 0; i < pre.length; i++) {
					pre[i] = svm.svm_predict(model, testprob.x[i]);
				}
				
				double counter = 0;
				for(int i = 0; i < pre.length; i++) {
					if(pre[i] == testprob.y[i]) {
						counter++;
					}
				}
				System.out.println("Accuracy = " + (counter / pre.length) );
			}
		}
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
