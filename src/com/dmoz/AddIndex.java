package com.dmoz;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;

public class AddIndex {

	public static void main(String[] args) throws IOException {
		// TODO 自动生成的方法存根
		String trainInFile = "F:\\DataSets\\NLPCC\\nram_word\\train.svm";
		String trainOutFile = "F:\\DataSets\\NLPCC\\nram_word\\train.svm.txt";
		String testInFile = "F:\\DataSets\\NLPCC\\nram_word\\test.svm";
		String testOutFile = "F:\\DataSets\\NLPCC\\nram_word\\test.svm.txt";
		addIndex(trainInFile, trainOutFile);
		addIndex(testInFile, testOutFile);
	}

	public static void addIndex(String inputfile, String outputfile) throws IOException {
		BufferedReader in = new BufferedReader(
				new InputStreamReader(new FileInputStream(inputfile)));
		BufferedWriter out = new BufferedWriter(
				new OutputStreamWriter(new FileOutputStream(outputfile)));
		String line = null;
		while((line = in.readLine()) != null) {
			String[] splits = line.split("\\s+|:|\n|\r|\t");
			line = new String();
			line += splits[0] + " ";
			for(int i = 1; i < splits.length; i+=2) {
				int index = Integer.parseInt(splits[i]) + 1;
				line += index + ":" + splits[i+1] + " ";
			}
			line += "\n";
			out.write(line);
		}
		in.close();
		out.close();
	}
}
