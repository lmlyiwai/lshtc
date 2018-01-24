package com.fileInputOutput;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import com.sparseVector.DataPoint;

/**
 * ѵ������������
 */
public class Problem {
    /**
     * ѵ����������
     */
    public int l;

    /**
     * ����ά��
     */
    public int n;

    /**
     * ����ǩ
     */
    public int[][] y;

    /**
     * ������������
     */
    public DataPoint[][] x;

    /**
     * bias
     */
    public double bias;

    /**
     * �����ļ�������ö�Ӧ���RCV1���ݼ�������
     */
    public void getFileLabels(Map<Integer, int[]> iamap) {
        int fileid = 0;
        for (int i = 0; i < y.length; i++) {
            fileid = y[i][0];
            y[i] = iamap.get(fileid);
        }
    }

    /**
     * ����ǰn�����������
     */
    public void smallProblem(int n) {
        this.l = n;
        DataPoint[][] ssample = new DataPoint[n][];
        int[][] sy = new int[n][];

        for (int i = 0; i < n; i++) {
            ssample[i] = this.x[i];
            sy[i] = this.y[i];
        }
        this.x = ssample;
        this.y = sy;
    }

    /**
     * �����������д���ļ�
     *
     * @throws IOException
     */
    public void writeSamplesToFile(Problem prob, String filename) throws IOException {
        BufferedWriter out = new BufferedWriter(
                new OutputStreamWriter(new FileOutputStream(filename)));
        String line = null;
        int n = prob.l;

        int[] label;
        DataPoint[] dp;
        int j;
        int k;
        for (int i = 0; i < n; i++) {
            dp = prob.x[i];
            label = prob.y[i];
            line = new String();
            for (j = 0; j < label.length; j++) {
                line += label[j];
                if (j < label.length - 1) {
                    line += ",";
                } else {
                    line += " ";
                }
            }

            for (k = 0; k < dp.length; k++) {
                line += dp[k].index + ":" + dp[k].value;
                line += " ";
            }

            line += "\n";
            out.write(line);
        }
        out.close();
    }

    /**
     * @throws IOException               ��ȡѵ������
     * @throws InvalidInputDataException
     */
    public static Problem readProblem(File file, double bias) throws IOException, InvalidInputDataException {
        BufferedReader fp = new BufferedReader(new FileReader(file));
        List<int[]> vy = new ArrayList<int[]>();
        List<DataPoint[]> vx = new ArrayList<DataPoint[]>();

        int max_index = 0;

        int lineNr = 0;

        try {
            while (true) {
                String line = fp.readLine();
                if (line == null) break;
                lineNr++;

                String[] st = line.split("\\s+|\t|\n|\r|\f|:");
                if (st.length <= 1) {
                    System.out.println("�������������ʽ");
                    return null;
                }

                String label = st[0];
                String[] labels = label.split(",");
                int[] labs = new int[labels.length];        //������Ӧ��ǩ
                for (int i = 0; i < labs.length; i++) {
                    labs[i] = Integer.parseInt(labels[i]);
                }
                vy.add(labs);

                int m = st.length / 2;
                DataPoint[] x;
                if (bias >= 0) {
                    x = new DataPoint[m + 1];
                } else {
                    x = new DataPoint[m];
                }

                lmlRefactor(file, lineNr, st, m, x);

                if (m > 0) {
                    max_index = Math.max(max_index, x[m - 1].index);
                }
                vx.add(x);
            }

            return constructProblem(vy, vx, max_index, bias);
        } finally {
            fp.close();
        }
    }

    private static void lmlRefactor(File file, int lineNr, String[] st, int m, DataPoint[] x) throws InvalidInputDataException {
        int indexBefore = 0;
        String token;
        for (int j = 0; j < m; j++) {
            token = st[2 * j + 1];
            token = token.trim();
            int index;
            try {
                index = Integer.parseInt(token);
            } catch (NumberFormatException e) {
                throw new InvalidInputDataException("��Ч��index:" + token, file, lineNr, e);
            }

            if (index < 0) throw new InvalidInputDataException("��Ч��index:" + index, file, lineNr);
            if (index <= indexBefore) throw new InvalidInputDataException("index Ӧ���Ե�����ʽ����", file, lineNr);
            indexBefore = index;

            token = st[2 * j + 2];
            try {
                double value = Double.parseDouble(token);
                x[j] = new DataPoint(index, value);
            } catch (NumberFormatException e) {
                throw new InvalidInputDataException("��Ч��value:" + token, file, lineNr);
            }
        }
    }

    /**
     *
     * */
    public static Problem constructProblem(List<int[]> vy, List<DataPoint[]> vx, int max_index, double bias) {
        Problem prob = new Problem();
        prob.bias = bias;
        prob.l = vy.size();
        prob.n = max_index;

        if (bias >= 0) {
            prob.n++;
        }
        prob.x = new DataPoint[prob.l][];
        for (int i = 0; i < prob.l; i++) {
            prob.x[i] = vx.get(i);
            if (bias >= 0) {
                assert prob.x[i][prob.x[i].length - 1] == null;
                prob.x[i][prob.x[i].length - 1] = new DataPoint(max_index + 1, bias);
            }
        }

        prob.y = new int[prob.l][];
        for (int i = 0; i < prob.l; i++) {
            prob.y[i] = vy.get(i);
        }

        return prob;
    }

    /**
     * ��ǰ����ȡn���������������Ϊ�յ�������
     */
    public Problem getFirstNSamples(int num) {
        Problem p = new Problem();
        p.bias = this.bias;
        p.l = num;
        p.n = this.n;
        p.y = new int[num][];
        p.x = new DataPoint[num][];

        int counter = 0;
        for (int i = 0; i < this.l; i++) {
            if (this.y[i].length != 0) {
                p.x[counter] = this.x[i];
                p.y[counter] = this.y[i];
                counter++;
            }

            if (counter >= num) {
                break;
            }
        }
        return p;
    }

    /**
     * �Ӻ���ǰȡnum������
     */
    public Problem getLastSamples(int num) {
        Problem p = new Problem();
        p.bias = this.bias;
        p.l = num;
        p.n = this.n;
        p.y = new int[num][];
        p.x = new DataPoint[num][];

        int counter = 0;
        for (int i = this.l - 1; i >= 0; i--) {
            if (this.y[i].length != 0) {
                p.x[counter] = this.x[i];
                p.y[counter] = this.y[i];
                counter++;
            }

            if (counter >= num) {
                break;
            }
        }
        return p;
    }

    /**
     * ��õ���������
     */
    public Problem getSingleLabelSamples() {
        int i;
        int counter = 0;
        for (i = 0; i < this.l; i++) {
            if (y[i].length == 1) {
                counter++;
            }
        }

        Problem newProb = new Problem();
        newProb.bias = bias;
        newProb.l = counter;
        newProb.n = n;
        newProb.x = new DataPoint[counter][];
        newProb.y = new int[counter][];

        counter = 0;
        for (i = 0; i < this.l; i++) {
            if (y[i].length == 1) {
                newProb.x[counter] = this.x[i];
                newProb.y[counter] = this.y[i];
                counter++;
            }
        }

        return newProb;
    }

    public static Problem getSmallProblem(Problem prob, int[] index) {
        Problem newProb = new Problem();
        newProb.l = index.length;
        newProb.n = prob.n;
        newProb.bias = prob.bias;

        newProb.x = new DataPoint[newProb.l][];
        newProb.y = new int[newProb.l][];

        for (int i = 0; i < index.length; i++) {
            newProb.x[i] = prob.x[index[i]];
            newProb.y[i] = prob.y[index[i]];
        }

        return newProb;
    }

    public void scale() {
        if (this.x == null) {
            return;
        }

        int i, j;
        double sumOfSquare;
        int tLength;
        for (i = 0; i < this.x.length; i++) {
            sumOfSquare = 0.0;
            if (bias > 0) {
                tLength = this.x[i].length - 1;
            } else {
                tLength = this.x[i].length;
            }

            for (j = 0; j < tLength; j++) {
                sumOfSquare += this.x[i][j].value * this.x[i][j].value;
            }

            for (j = 0; j < tLength; j++) {
                this.x[i][j].value = this.x[i][j].value / sumOfSquare;
            }
        }
    }

    /**
     * ��ȡѵ����������ǩ���ַ�����ʽ��������ת��Ϊָ�����
     */
    public static Problem readProblem(File file, double bias, Map<String, Integer> map) throws IOException, InvalidInputDataException {
        BufferedReader fp = new BufferedReader(new FileReader(file));
        List<int[]> vy = new ArrayList<int[]>();
        List<DataPoint[]> vx = new ArrayList<DataPoint[]>();

        int max_index = 0;

        int lineNr = 0;

        try {
            while (true) {
                String line = fp.readLine();
                if (line == null) break;
                lineNr++;

                String[] st = line.split("\\s+|\t|\n|\r|\f|:");
                if (st.length <= 1) {
                    System.out.println("�������������ʽ");
                    return null;
                }

                String label = st[0];
                String[] labels = label.split(",");
                int[] labs = new int[labels.length];        //������Ӧ��ǩ
                for (int i = 0; i < labs.length; i++) {
//					labs[i] = Integer.parseInt(labels[i]);
                    labs[i] = map.get(labels[i].trim());
                }
                vy.add(labs);

                int m = st.length / 2;
                DataPoint[] x;
                if (bias >= 0) {
                    x = new DataPoint[m + 1];
                } else {
                    x = new DataPoint[m];
                }

                lmlRefactor(file, lineNr, st, m, x);

                if (m > 0) {
                    max_index = Math.max(max_index, x[m - 1].index);
                }
                vx.add(x);
            }

            return constructProblem(vy, vx, max_index, bias);
        } finally {
            fp.close();
        }
    }

    /**
     * Ϊÿ��·��ָ��һ����ֵ��ʶ
     */
    public static Map<String, Integer> getMap(int[][] y) {
        return null;
    }
}
