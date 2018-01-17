package com.tools;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.List;

public class ITF {
	public static void itf(String inputfile, String outputfile, boolean containZero) throws IOException {
		BufferedReader in = new BufferedReader(
				new InputStreamReader(new FileInputStream(inputfile)));
		BufferedWriter out = new BufferedWriter(
				new OutputStreamWriter(new FileOutputStream(outputfile)));
		String line = null;
		String[] splits = null;
		while((line = in.readLine()) != null) {
			splits = line.split("\\s|:|\r|\n|\t");
			line = new String();
			line += splits[0] + " ";
			int i = 1;
			if(containZero) {
				i = 3;
			}
			while(i < splits.length) {
				String key = splits[i];
				String value = splits[i+1];
				int v = Integer.parseInt(value);
				double itf= 1.0 - (1.0 / (v + 1.0));
				line += key + ":" + itf;
				if(i < splits.length - 2) {
					line += " ";
				} else {
					line += "\n";
				}
				i += 2;
			}
			out.write(line);
		}
		in.close();
		out.close();
	}
	
	public static void itfNorm(String inputfile, String outputfile, boolean containZero) throws IOException {
		BufferedReader in = new BufferedReader(
				new InputStreamReader(new FileInputStream(inputfile)));
		BufferedWriter out = new BufferedWriter(
				new OutputStreamWriter(new FileOutputStream(outputfile)));
		String line = null;
		String[] splits = null;
		List<String> key = new ArrayList<String>();
		List<Double> value = new ArrayList<Double>();
		double sum = 0.0;
		while((line = in.readLine()) != null) {
			splits = line.split("\\s|:|\r|\n|\t");
			line = new String();
			line += splits[0] + " ";
			int i = 1;
			if(containZero) {
				i = 3;
			}
			sum = 0.0;
			while(i < splits.length) {
				String k = splits[i];
				String val = splits[i+1];
				int v = Integer.parseInt(val);
				double itf= 1.0 - (1.0 / (v + 1.0));
				sum += itf * itf;
				key.add(k);
				value.add(itf);
				i += 2;
			}
			for(int j = 0; j < key.size(); j++) {
				line += key.get(j);
				line += ":";
				line += value.get(j) / Math.sqrt(sum);
				if(j < key.size() - 1) {
					line += " ";
				} else {
					line += "\n";
				}
			}
			key.clear();
			value.clear();
			out.write(line);
		}
		in.close();
		out.close();
	}
}
