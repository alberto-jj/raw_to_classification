import base64
from io import BytesIO
import json

# Get the derivatives path in BIDS format
def get_derivative_path(layout,eeg_file,output_entity,suffix,output_extension,bids_root,derivatives_root):
    entities = layout.parse_file_entities(eeg_file)
    derivative_path = eeg_file.replace(bids_root,derivatives_root)
    derivative_path = derivative_path.replace(entities['extension'],'')
    derivative_path = derivative_path.split('_')
    desc = 'desc-' + output_entity
    derivative_path = derivative_path[:-1] + [desc] + [suffix]
    derivative_path = '_'.join(derivative_path) + output_extension 
    return derivative_path

# Save FIGS
def save_figs_in_html(htmlfile,figures):
    htmls = ['<img src=\'data:image/png;base64,{}\'>'.format(imgfoo(fig)) for fig in figures]
    html = "\n".join(htmls)
    with open(htmlfile,'w') as f:
        f.write(html)

def imgfoo(fig):
    tmpfile = BytesIO()
    fig.savefig(tmpfile, format='png')
    encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
    return encoded

def save_dict_to_json(jsonfile,data):
    with open(jsonfile, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def parse_bids(bidsname,key):
    entities=bidsname.split('_')
    suffix = entities[-1]
    ext = suffix.split('.')[-1]
    suffix = suffix.split('.')[0]
    entities = entities[:-1]
    d={}
    for item in entities:
        l=item.split('-')
        key=l[0]
        val=l[1]
        d[key]=val
    d['suffix']=suffix
    return d
