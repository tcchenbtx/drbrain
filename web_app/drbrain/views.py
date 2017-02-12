import os
from flask import render_template, request, Flask, redirect, url_for, send_from_directory
from werkzeug import secure_filename
from drbrain import app
#from sqlalchemy import create_engine
#from sqlalchemy_utils import database_exists, create_database
import pandas as pd
#import psycopg2
from a_Model import ModelIt
import subprocess
import nibabel as nib
from matplotlib import pyplot as plt
import re
from brain_anomaly_check import run_brain_anomaly_2d, run_brain_anomaly
from go_classify import go_predict
import numpy as np



# for upload

# absolute path
app.config['APP_ROOT'] = os.path.dirname(os.path.abspath(__file__))
# This is the path to the upload directory
app.config['UPLOAD_FOLDER'] = os.path.join(app.config['APP_ROOT'], "uploads")
# These are the extension that we are accepting to be uploaded
app.config['ALLOWED_EXTENSIONS'] = set(['nii', 'nii.gz'])
app.config['STATIC_FOLDER'] = os.path.join(app.config['APP_ROOT'], "static")
app.config['OUTPUT_FOLDER'] = os.path.join(app.config['STATIC_FOLDER'], "output")
app.config['OUTPUT_PROB_FOLDER'] = os.path.join(app.config['STATIC_FOLDER'], "output_prob")


def go_cmd1(target,output):
    cmd = ["fsl5.0-fslreorient2std","%s" % target, "%s" % output]
    p = subprocess.Popen(cmd, stdout = subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            stdin=subprocess.PIPE)
    out,err = p.communicate()
    print("cmd1 done")
    return

def go_cmd2(out1, out2):
    cmd = ["fsl5.0-robustfov", "-i", "%s" % out1, "-r", "%s" % out2]
    p = subprocess.Popen(cmd, stdout = subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            stdin=subprocess.PIPE)
    out,err = p.communicate()
    print("cmd2 done")
    return

def go_cmd3(out1, out2):
    cmd = ["fsl5.0-bet", "%s" % out1, "%s" % out2]
    p = subprocess.Popen(cmd, stdout = subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            stdin=subprocess.PIPE)
    out,err = p.communicate()
    print("cmd3 done")
    return

def go_cmd4(out1, out2, out3):
    cmd = ["fsl5.0-flirt", "-interp", "spline", "-dof", "12", "-in", "%s" % out1, "-ref", "%s" % out2, "-dof", "12", "-out", "%s" % out3]
    p = subprocess.Popen(cmd, stdout = subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            stdin=subprocess.PIPE)
    out,err = p.communicate()
    print("cmd4 done")
    return


# For a given file, return whether it's an allowed type or not
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']



# user = 'tzuchieh' #add your username here (same as previous postgreSQL)
# host = 'localhost'
# dbname = 'birth_db'
# db = create_engine('postgres://%s@%s/%s'%(user,host,dbname))
# con = None
# con = psycopg2.connect(database = dbname, user = user)

@app.route('/')
def index():
    return render_template("index.html")

# @app.route('/index')
# def index():
#     return render_template("index.html",
#        title = 'Home', user = { 'nickname': 'Miguel' },
#        )

# @app.route('/volume-viewer-demo.html')
# def volume_demo():
#     return render_template("volume-viewer-demo.html")

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      f = request.files['file']
      f.save(secure_filename(f.filename))
      return 'file uploaded successfully'


@app.route('/learn_more')
def show_slide():
    return render_template("learn_more.html")



@app.route('/db')
def birth_page():
    sql_query = """                                                                       
                SELECT * FROM birth_data_table WHERE delivery_method='Cesarean';          
                """
    query_results = pd.read_sql_query(sql_query,con)
    births = ""
    for i in range(0,10):
        births += query_results.iloc[i]['birth_month']
        births += "<br>"
    return births


@app.route('/db_fancy')
def cesareans_page_fancy():
    sql_query = """
               SELECT index, attendant, birth_month FROM birth_data_table WHERE delivery_method='Cesarean';
                """
    query_results=pd.read_sql_query(sql_query,con)
    births = []
    for i in range(0,query_results.shape[0]):
        births.append(dict(index=query_results.iloc[i]['index'], attendant=query_results.iloc[i]['attendant'], birth_month=query_results.iloc[i]['birth_month']))
    return render_template('cesareans.html',births=births)

@app.route('/input')
def cesareans_input():
    return render_template("input.html")


@app.route('/goanalysis', methods=['POST'])
def upload():
    # Get the name of the uploaded file
    select = request.form.get('Go_select')
    print(select)

    if select == "normal":
        file = True
        target_file_path = os.path.join(app.config['UPLOAD_FOLDER'], "ADNI_111_S_1111_S11111.nii")
        filename = "ADNI_111_S_1111_S11111.nii"
    elif select == "ad":
        file = True
        target_file_path = os.path.join(app.config['UPLOAD_FOLDER'], "ADNI_222_S_2222_S22222.nii")
        filename = "ADNI_222_S_2222_S22222.nii"
    elif select == "pd":
        file = True
        target_file_path = os.path.join(app.config['UPLOAD_FOLDER'], "ADNI_333_S_3333_S33333.nii")
        filename = "ADNI_333_S_3333_S33333.nii"
    elif select == "user_choose":
        file = request.files['user_input']
        if not allowed_file(file.filename):
            return render_template('file_error.html')
        else:
            filename = secure_filename(file.filename)
            print("filename:")
            print(filename)
            target_file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(target_file_path)
    else:
        file = None


    if not file:
        return render_template('file_error.html')

    # Check if the file is one of the allowed types/extensions
   # if file and allowed_file(file.filename):
        # Make the filename safe, remove unsupported chars

        # Move the file form the temporal folder to
        # the upload folder we setup



    file_pattern = re.compile(r'(.*)\.nii$|(.*)\.nii\.gz$')
    unique_filename_check = re.findall(file_pattern, filename)
    for i in unique_filename_check[0]:
        if i:
            unique_filename = i

    print("unique_filename:")
    print(unique_filename)
    print("target_file_path")
    print(target_file_path)

    ref_brain_file = os.path.join(app.config["STATIC_FOLDER"], "MNI152_T1_2mm_brain.nii.gz")
    output_reorient_file = os.path.join(app.config['UPLOAD_FOLDER'], "%s_reorient.nii" % unique_filename)
    print(output_reorient_file)
    output_crop_file = os.path.join(app.config['UPLOAD_FOLDER'], "%s_crop.nii" % unique_filename)
    output_brainonly_file = os.path.join(app.config['UPLOAD_FOLDER'], "%s_brainonly.nii" % unique_filename)
    output_stdbrain_file = os.path.join(app.config['UPLOAD_FOLDER'], "%s_stdbrain.nii.gz" % unique_filename)
    
    #subprocess.call("fsl5.0-fslreorient2std %s %s" % (target_file_path, output_reorient_file), shell=True)
    cmd1 = ["fsl5.0-fslreorient2std","%s" % target_file_path, "%s" % output_reorient_file]
    cmd2 = ["fsl5.0-robustfov", "-i", "%s" % output_reorient_file, "-r", "%s" % output_crop_file]
    cmd3 = ["fsl5.0-bet", "%s" % output_crop_file, "%s" % output_brainonly_file]
    cmd4 = ["fsl5.0-flirt", "-interp", "spline", "-dof", "12", "-in", "%s" % output_brainonly_file, "-ref", "%s" % ref_brain_file, "-dof", "12", "-out", "%s" % output_stdbrain_file]
    
    ocmd1 = "fsl5.0-fslreorient2std %s %s" % (target_file_path, output_reorient_file)
    ocmd2 = "fsl5.0-robustfov -i %s -r %s" %(output_reorient_file, output_crop_file)
    ocmd3 = "fsl5.0-bet", "%s" % output_crop_file, "%s" % output_brainonly_file
    ocmd4 = "fsl5.0-flirt", "-interp", "spline", "-dof", "12", "-in", "%s" % output_brainonly_file, "-ref", "%s" % ref_brain_file, "-dof", "12", "-out", "%s" % output_stdbrain_file
    

    go_cmd1(target_file_path, output_reorient_file)
    go_cmd2(output_reorient_file, output_crop_file)
    go_cmd3(output_crop_file, output_brainonly_file)
    go_cmd4(output_brainonly_file, ref_brain_file, output_stdbrain_file)
    #all_cmd2 = [ocmd1, ocmd2, ocmd3, ocmd4]
    #all_cmd = [cmd1, cmd2, cmd3, cmd4]

    #for cmd in all_cmd2:
    #    subprocess.Popen(cmd, shell=True)
    #subprocess.Popen(cmd1, stdout = subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE)
    #subprocess.call("fsl5.0-robustfov -i %s -r %s" %(output_reorient_file, output_crop_file), shell=True)
    #cmd = ["fsl5.0-robustfov", "-i", "%s" % output_reorient_file, "-r", "%s" % output_crop_file]
    #subprocess.Popen(cmd2, stdout = subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE)

    #subprocess.call("fsl5.0-bet %s %s" %(output_crop_file, output_brainonly_file), shell=True)
    #cmd = ["fsl5.0-bet", "%s" % output_crop_file, "%s" % output_brainonly_file]
    #subprocess.Popen(cmd3, stdout = subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE)    

    #subprocess.call("fsl5.0-flirt -interp spline -dof 12 -in %s -ref %s -dof 12 -out %s\n" % (output_brainonly_file, ref_brain_file, output_stdbrain_file), shell=True)
    #cmd = ["fsl5.0-flirt", "-interp", "spline", "-dof", "12", "-in", "%s" % output_brainonly_file, "-ref", "%s" % ref_brain_file, "-dof", "12", "-out", "%s" % output_stdbrain_file]
    #subprocess.Popen(cmd)
    #subprocess.Popen(cmd4, stdout = subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE)

    print("Pass!!")
    # run command to make figure?
    nii_load = nib.load(output_reorient_file)
    nii_data = nii_load.get_data()

    if len(nii_data.shape) == 4:
        new_shape = nii_data.shape[:-1]
        nii_data = nii_data.reshape(new_shape)


    beauty_nii = nib.load(output_stdbrain_file)
    beauty_nii_data = beauty_nii.get_data()
    print ("to make image")


    f, ax = plt.subplots(1, 3, figsize=(2,3))
    plt.setp(ax, xticks=[], xticklabels=[], yticks=[])
    ax[0].imshow(nii_data[int(nii_data.shape[0]/2), :, :], cmap='gray')
    ax[1].imshow(nii_data[:, int(nii_data.shape[1]/2), :], cmap='gray')
    ax[2].imshow(nii_data[:, :, int(nii_data.shape[2]/2)], cmap='gray')
    plt.tight_layout()

    output_fig_path = os.path.join(app.config['STATIC_FOLDER'], "output","%s_structure_1.png" % unique_filename)
    plt.savefig(output_fig_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()


    f, ax = plt.subplots(1, 3, figsize=(2,3))
    plt.setp(ax, xticks=[], xticklabels=[], yticks=[])
    ax[0].imshow(beauty_nii_data[int(beauty_nii_data.shape[0]/2), :, :], cmap='gray')
    ax[1].imshow(beauty_nii_data[:, int(beauty_nii_data.shape[1]/2), :], cmap='gray')
    ax[2].imshow(beauty_nii_data[:, :, int(beauty_nii_data.shape[2]/2)], cmap='gray')
    plt.tight_layout()
    output_fig_path = os.path.join(app.config['STATIC_FOLDER'], "output","%s_structure_2.png" % unique_filename)
    plt.savefig(output_fig_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()


    #f, ax = plt.subplots(2, 3, figsize=(4,3))
    #plt.setp(ax, xticks=[], xticklabels=[], yticks=[])
    #ax[0, 0].imshow(nii_data[int(nii_data.shape[0]/2), :, :], cmap='gray')
    #ax[0, 1].imshow(nii_data[:, int(nii_data.shape[1]/2), :], cmap='gray')
    #ax[0, 2].imshow(nii_data[:, :, int(nii_data.shape[2]/2)], cmap='gray')
    #ax[1, 0].imshow(beauty_nii_data[int(beauty_nii_data.shape[0]/2), :, :], cmap='gray')
    #ax[1, 1].imshow(beauty_nii_data[:, int(beauty_nii_data.shape[1]/2), :], cmap='gray')
    #ax[1, 2].imshow(beauty_nii_data[:, :, int(beauty_nii_data.shape[2]/2)], cmap='gray')
    
    #print("to save image")
    #print(os.path.join(app.config['STATIC_FOLDER'], "output","%s_structure.png" % unique_filename))
    #output_fig_path = os.path.join(app.config['STATIC_FOLDER'], "output","%s_structure.png" % unique_filename)
    #plt.savefig(output_fig_path)
    #plt.close()







    # Redirect the user to the uploaded_file route, which
    # will basically show on the browser the uploaded file

    print(unique_filename)
    to_show_fig = unique_filename

        # run anomaly check
    anomaly_check_outcome = run_brain_anomaly(beauty_nii_data)
    print(anomaly_check_outcome)

    if anomaly_check_outcome[0] != 1:
        bad = "Bad"
        get_class = "Alzheimer's Disease"
        to_show_prob = "mini"

    else:
        bad = "Good"
        ## apply classification
        #output = np.array([[0.143, 0.756, 0.003]])  ########## apply classification
        output = go_predict(beauty_nii_data)
        print(output)
        prob_list = output.tolist()
        prob_list = prob_list[0]
        objects = ('Normal', 'AD', 'PD')
        y_pos = np.arange(len(objects))

        output_class = np.argmax(output)
        if output_class == 0:
            get_class = "Normal healthy"
        elif output_class == 1:
            get_class = "Alzheimer's Disease"
        elif output_class == 2:
            get_class = "Parkinson's Disease"
        else:
            get_class = "I cannot predict!?"
        
        print(output_class)
        most_prob = prob_list[output_class]
        print(most_prob)
        report_prob = "%.1f%%" % (most_prob*100)
        print(report_prob)
        # get_class = "Alzheimer's Disease"  ############# for now!!

        print("make graph")
        plt.style.use('ggplot')
        plt.bar(y_pos, prob_list, align='center', alpha=0.8)
        plt.ylim(0, 1.0)
        plt.xticks(y_pos, objects, fontsize=15)
        plt.yticks(fontsize=15)
        plt.ylabel('Probability', fontsize=20)
        plt.title("Predict as %s with probability of %s" % (get_class, report_prob))

        profig_output = os.path.join(app.config['OUTPUT_PROB_FOLDER'],"%s_prob.png" % unique_filename)
        plt.savefig(profig_output)
        plt.close()

        to_show_prob = unique_filename + '_prob.png'
        print(to_show_prob)

        # url for a defined function, also direct filename parameter
        #return redirect(url_for('uploaded_file', filename=unique_filename))
        # return redirect(url_for('uploaded_file', filename=unique_filename))
    return render_template('index.html', outfilename=to_show_fig, section='result', outlier=bad, output_class=get_class, prob_fig=to_show_prob)


# @app.route('/#services')
# def uploaded_file():
#     filename = request.args.get('input')
#     outputfile = os.path.join(app.config['STATIC_FOLDER'], "output", "%s_structure.png" % filename)
#     print(outputfile)
#     return render_template('index.html', outputfilepath=outputfile, section='services')
#
#     # return send_from_directory(app.config['UPLOAD_FOLDER'],
#     #                            filename)
#
# @app.route('/viewer')
# def go_viewer():
#     return render_template("volume-viewer-demo.html")
#
#     # return send_from_directory(app.config['UPLOAD_FOLDER'],
#     #                            filename)



# @app.route('/output')
# def cesareans_output():
#   #pull 'birth_month' from input field and store it
#     patient = request.args.get('birth_month')
#     #just select the Cesareans  from the birth dtabase for the month that the user inputs
#     query = "SELECT index, attendant, birth_month FROM birth_data_table WHERE delivery_method='Cesarean' AND birth_month='%s'" % patient
#     print query
#     query_results=pd.read_sql_query(query,con)
#     print query_results
#     births = []
#     for i in range(0,query_results.shape[0]):
#         births.append(dict(index=query_results.iloc[i]['index'], attendant=query_results.iloc[i]['attendant'], birth_month=query_results.iloc[i]['birth_month']))
#         the_result = ModelIt(patient,births)
#     return render_template("output.html", births = births, the_result = the_result)
#
