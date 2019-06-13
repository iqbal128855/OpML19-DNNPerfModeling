import logging
import socket
import os
import sys
import json 
from datetime import datetime
from optparse import OptionParser
from Configuration import Config as cfg
from Src.GenerateParams import GenerateParams
from Src.SetWorkload import SetWorkload
from Visualization.GenerateFigure import GenerateFigure

def config_option_parser():
    """This function is used to configure option parser 
    @returns:
        options: option parser handle
    """
    usage="""USAGE: %python RunCOnfigLabyrinth.py -m [model] -r [row] -c [col]
             Model ResNet50:                  python RunConfigLabyrinth.py -o resnet50 -r 224 -c 224
             Model VGG16:                     python RunConfigLabyrinth.py -o vgg16 -r 224 -c 224
             Model VGG19:                     python RunConfigLabyrinth.py -o vgg19 -r 224 -c 224
             Model Xception:                  python RunConfigLabyrinth.py -o xception -r 224 -c 224
             Model InceptionV3:               python RunConfigLabyrinth.py -o inceptionv3 -r 224 -c 224
    """
    parser=OptionParser(usage=usage)
    parser.add_option('-m', "--model",
                      action="store",
                      type="string",
                      dest="mod",
                      help="Type of Model")
    parser.add_option('-r', "--rows",
                      action="store",
                      type="int",
                      dest="rows",
                      help="rows")
    parser.add_option('-c', "--columns",
                      action="store",
                      type="int",
                      dest="cols",
                      help="columns")
    (options,args)=parser.parse_args()
    return (options, usage)

def config_logger():
    """This function is used to configure logging information
    @returns: 
        logger: logging object
    """
    # get log directory 
    log_dir=os.getcwd()+cfg.log_dir
    log_file_name="logfile_{0}".format(str(datetime.now().date()))
    log_file=os.path.join(log_dir,log_file_name)
    
    # get logger object
    ip=socket.gethostbyname(socket.gethostname())
    extra={"ip_address":ip}
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logger=logging.getLogger(__name__)
    hdlr = logging.FileHandler(log_file)
    
    # define log format
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(ip_address)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    
    # define log level
    logger.setLevel(logging.INFO)
    logger = logging.LoggerAdapter(logger, extra)
    logger.info("[STATUS]: Start RunTest")
    return logger
                                
if __name__=="__main__":
    logger=config_logger()
    (options,usage)=config_option_parser()
    # workload model name
    model_name=str(options.mod)
    rows=options.rows
    cols=options.cols    
    
    # get input data 
    input_dir="{0}{1}".format(os.getcwd(), str(cfg.input_dir))
    input_data=[f for f in os.listdir(input_dir) if ((os.path.isfile(os.path.join(input_dir,f))) and (os.stat(os.path.join(input_dir,f)).st_size > 0))]
    swl=SetWorkload(logger,
                    model_name,
                    rows,
                    cols)
 
    for sample in input_data:
        sample=str(sample)
        data="{0}{1}".format(input_dir,sample)
        (size,
        test_data, 
        model)=swl.get_workload_params(data)
        GP=GenerateParams(logger,
                          model,
                          test_data,
                          model_name,
                          sample,
                          str(size))
              
    #gf=GenerateFigure(logger, "TX2") 
