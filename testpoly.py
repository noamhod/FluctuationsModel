import ROOT
import array

ROOT.gROOT.SetBatch(1)
ROOT.gStyle.SetOptFit(0)
ROOT.gStyle.SetOptStat(0)

ROOT.gStyle.SetPadBottomMargin(0.13)
ROOT.gStyle.SetPadLeftMargin(0.11)
ROOT.gStyle.SetPadRightMargin(0.06)
ROOT.gStyle.SetPadTopMargin(0.07)

def is_exposed(histo, ix, iy, direction):
    """
    Check if the side of the bin in the specified direction is exposed.
    Directions: 
    - 'left': check if the bin to the left is 0 or out of bounds.
    - 'right': check if the bin to the right is 0 or out of bounds.
    - 'bottom': check if the bin below is 0 or out of bounds.
    - 'top': check if the bin above is 0 or out of bounds.
    """
    Nx = histo.GetNbinsX()
    Ny = histo.GetNbinsY()
    if(direction=='left'):     return (ix==1  or histo.GetBinContent(ix-1, iy)<1)
    elif(direction=='right'):  return (ix==Nx or histo.GetBinContent(ix+1, iy)<1)
    elif(direction=='bottom'): return (iy==1  or histo.GetBinContent(ix, iy-1)<1)
    elif(direction=='top'):    return (iy==Ny or histo.GetBinContent(ix, iy+1)<1)
    return False

def get_exposed_edges(histo, ix, iy, Nx, Ny):
    """
    Get the exposed edges of a boundary bin (ix, iy).
    Return a list of line segments representing the exposed edges.
    Each line segment is defined by two (x, y) points.
    """
    xlow = histo.GetXaxis().GetBinLowEdge(ix)
    xup  = histo.GetXaxis().GetBinUpEdge(ix)
    ylow = histo.GetYaxis().GetBinLowEdge(iy)
    yup  = histo.GetYaxis().GetBinUpEdge(iy)
    edges = []
    # Check each of the four sides of the bin and add exposed edges
    if(is_exposed(histo, ix, iy, 'left')):   edges.append((xlow, ylow, xlow, yup))  # Left edge
    if(is_exposed(histo, ix, iy, 'right')):  edges.append((xup,  ylow, xup,  yup))  # Right edge
    if(is_exposed(histo, ix, iy, 'bottom')): edges.append((xlow, ylow, xup,  ylow))  # Bottom edge
    if(is_exposed(histo, ix, iy, 'top')):    edges.append((xlow, yup,  xup,  yup))  # Top edge
    return edges

def find_contour(histo):
    """
    Construct a TH2Poly object that represents the outer contour of the cluster.
    Only exposed edges of boundary bins are included in the contour.
    """
    Nx = histo.GetNbinsX()
    Ny = histo.GetNbinsY()
    poly = ROOT.TH2Poly()
    for ix in range(1,Nx+1):
        for iy in range(1,Ny+1):
            if(histo.GetBinContent(ix,iy)>0):  # Cluster bin
                edges = get_exposed_edges(histo, ix, iy, Nx, Ny)
                for edge in edges:
                    # Unpack the edge (x1, y1, x2, y2)
                    x1, y1, x2, y2 = edge
                    # Create arrays of x and y points for the edge (two points per edge)
                    x = array.array('d', [x1, x2])
                    y = array.array('d', [y1, y2])
                    poly.AddBin(2, x, y)  # Add the edge as a polygonal line
    
    # channge line properties
    poly_bins = poly.GetBins()
    for i in range(len(poly_bins)):
        gr = poly_bins[i].GetPolygon() ## TGraph
        gr.SetLineColor(ROOT.kBlack)
        # gr.SetLineStyle(1)
        # gr.SetLineWidth(2)
    
    return poly

# Example usage:
# Nx = 10
# Ny = 10
# Xmin, Xmax = 0, 10
# Ymin, Ymax = 0, 10
#
# histo = ROOT.TH2D("histo", "Cluster", Nx, Xmin, Xmax, Ny, Ymin, Ymax)
#
# # Fill the histogram with a simple cluster (a rectangular shape)
# for ix in range(4, 8):  # x from bin 4 to 7 (i.e., 3 <= x < 8)
#     for iy in range(3, 7):  # y from bin 3 to 6 (i.e., 2 <= y < 7)
#         histo.SetBinContent(ix, iy, 1)
# histo.SetBinContent(8,6,1)
# histo.SetBinContent(9,6,1)
# histo.SetBinContent(10,6,1)
# histo.SetBinContent(6,7,1)
# histo.SetBinContent(7,8,1)
# histo.SetBinContent(5,2,1)
# histo.SetBinContent(5,2,1)

tf = ROOT.TFile("regions_example.root","READ")

histos = {"BEBL":tf.Get("BEBL"), "TGAU":tf.Get("TGAU"), "IONBxEX1BxIONG":tf.Get("IONBxEX1BxIONG"), "IONBxIONGxEX1G":tf.Get("IONBxIONGxEX1G"), "IONBxEX1B":tf.Get("IONBxEX1B")}
hcolrs = {"BEBL":ROOT.kGray+2,   "TGAU":ROOT.kGreen+2,  "IONBxEX1BxIONG":ROOT.kWhite,             "IONBxIONGxEX1G":ROOT.kMagenta,            "IONBxEX1B":ROOT.kRed}

### Find the cluster contour and create a TH2Poly object
contours = {}

leg = ROOT.TLegend(0.13,0.15,0.5,0.40)
leg.SetFillStyle(4000) # will be transparent
leg.SetFillColor(0)
leg.SetTextFont(42)
leg.SetBorderSize(0)
for hname,hist in histos.items():
    contour = find_contour(hist)
    contours.update({hname:contour})
    hist.SetFillColorAlpha(hcolrs[hname],0.15)
    leg.AddEntry(hist,hname.replace("x","#otimes"),"f")
    

distribution = tf.Get("h_dL_vs_E")

# Draw the original histogram and the contour
canvas = ROOT.TCanvas()
canvas.SetTicks(1,1)
distribution.GetXaxis().SetTitleOffset(1.3)
distribution.GetYaxis().SetTitleOffset(1.1)
canvas.SetLogx()
canvas.SetLogy()
canvas.SetLogz()
distribution.Draw("col")
for hname,hist    in histos.items(): hist.Draw("box same")
for pname,contour in contours.items(): contour.Draw("L same")
leg.Draw("same")
canvas.RedrawAxis()
canvas.SaveAs("contour_clean.png")
canvas.SaveAs("Regions.pdf")