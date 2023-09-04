import re
import gzip


def load_amr_entries(fname, strip_comments=True):
    if fname.endswith('.gz'):
        with gzip.open(fname, 'rb', encoding='utf-8') as f:
            data = f.read().decode()
    else:
        with open(fname, encoding='utf-8') as f:
            data = f.read()

    entries = data.split('\n\n')            # split via standard amr
    entries = [e.strip() for e in entries]  # clean-up line-feeds, spaces, etc
    entries = [e for e in entries if e]     # remove any empty entries
    return entries

def load_indo_news_amr_entries(fname):
    if fname.endswith('.gz'):
        with gzip.open(fname, 'rb', encoding='utf-8') as f:
            data = f.read().decode()
    else:
        with open(fname, encoding='utf-8') as f:
            data = f.read()
    
    lines =  []
    check = False
    for l in data.splitlines():
        if(l.startswith("# ::id")): # Kompas and tempo used # ::id 
            check = True
            lines.append(l)
        elif(l == '' and check == True):
            check = False
        else:
            lines.append(l)
    # Strip off non-amr header info (see start of Little Prince corpus)
    # if strip_comments:
    #     lines = [l for l in data.splitlines() if not (l.startswith('#') and not \
    #              l.startswith('# ::'))]
    data = '\n'.join(lines)

    entries = data.split('\n\n')            # split via standard amr
    entries = [e.strip() for e in entries]  # clean-up line-feeds, spaces, etc
    entries = [e for e in entries if e]     # remove any empty entries
    return entries

def load_simple_amr_indonesia(fname):
    if fname.endswith('.gz'):
        with gzip.open(fname, 'rb', encoding='utf-8') as f:
            data = f.read().decode()
    else:
        with open(fname, encoding='utf-8') as f:
            data = f.read()
    
    lines =  []
    check = 0
    for l in data.splitlines():
        l = l.strip(" ")
        if(l.startswith("# ::id") or l.startswith("# ::snt")):
            check = 0
            lines.append(l)
        elif(l == '' and check == 0):
            continue
        elif(l == '' and check > 0) :
            lines.append(l)
        else:
            check += 1
            lines.append(l)

    data = '\n'.join(lines)

    entries = data.split('\n\n')            # split via standard amr
    entries = [e.strip() for e in entries]  # clean-up line-feeds, spaces, etc
    entries = [e for e in entries if e]     # remove any empty entries
    return entries

# Split the entry into graph lines and metadata lines
# note that line-feeds are stripped
def split_amr_meta(entry):
    meta_lines  = []
    graph_lines = []
    for line in entry.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith('# ::'):
            meta_lines.append(line)
        elif line.startswith('#'):
            continue
        else:
            graph_lines.append(line)
    return meta_lines, graph_lines


# Load the AMR file and return a dict "entries" with 2 keys, sent and graph
# sent and graph are both a list, string for each entry in the file
def load_amr_graph_sent(fpath):
    entries = {'sent':[], 'graph':[]}
    with open(fpath) as f:
        data = f.read()
    for entry in data.split('\n\n'):
        sent     = None
        gstrings = []
        for line in entry.splitlines():
            line = line.strip()
            if line.startswith('# ::snt'):
                sent = line[len('# ::snt'):].strip()
            if not line.startswith('#'):
                gstrings.append( line )
        if sent and gstrings:
            entries['sent'].append(sent)
            entries['graph'].append(' '.join(gstrings))
    return entries


# Strip all the metadata from the graph
# if one_line is True, return a single line with all extra spaces stripped
def get_graph_only(entry, one_line=False):
    graph_lines = []
    for line in entry.splitlines():
        if not line or line.startswith('#'):
            continue
        else:
            graph_lines.append(line)
    if one_line:
        graph_lines = [l.strip() for l in graph_lines]
        gstring     = ' '.join(graph_lines)
        gstring     = re.sub(' +', ' ', gstring) # squeeze multiple spaces into a single
        return gstring
    else:
        return '\n'.join(graph_lines)
