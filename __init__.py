import numpy, string
import random, math
import pylab as p
from mpl_toolkits.basemap import Basemap
import griddata.griddata as mgriddata

# Version 0.2
'''
for batch mode use a -dAGG option on the command line
'''
class grid():
    def __init__(self, *x):
        """Initialize an ascii grid. If one arguement then read grid from file else create new grid using new function."""
        if len(x)==1:
            self.read( x[0] )
        elif 4<=len(x)<=6:
            apply(self.new, x)

    def read( self, Filename ):
        """Read an ascii grid from a file"""
        try:
            self.name = Filename
            Filedata = open(self.name,'r').readlines()
            self.ncols     = string.atoi( Filedata[0].strip().split()[-1] )
            self.nrows     = string.atoi( Filedata[1].strip().split()[-1] )
            self.xllcorner = string.atof( Filedata[2].strip().split()[-1] )
            self.yllcorner = string.atof( Filedata[3].strip().split()[-1] )
            self.cellsize  = string.atof( Filedata[4].strip().split()[-1] )
            self.nodata    = string.atof( Filedata[5].strip().split()[-1] )
            self.data = numpy.ones( (self.nrows, self.ncols ) ) *1.0
            row = self.nrows
            for t in Filedata[6:]:
                row -= 1
                col = -1
                values = map(string.atof, t.strip().split())
                for x in values:
                    col += 1
                    self.data[(row,col)] = x
        except:
            print "Error opening grid ::", Filename
            raise

    def write( self, NewFilename='', Integer=True ):
        """Write and ascii grid to disk using either the name attribute or arguement passed in as filename"""
        try:
            if NewFilename != '':
                self.name=NewFilename
            Output = open( self.name, 'w' )
            Output.write( 'ncols\t\t %d\n' % self.ncols )
            Output.write( 'nrows\t\t %d\n' % self.nrows )
            Output.write( 'xllcorner\t\t %f\n' % self.xllcorner)
            Output.write( 'yllcorner\t\t %f\n' % self.yllcorner)
            Output.write( 'cellsize\t\t %f\n' % self.cellsize)
            if Integer:
                Output.write( 'NODATA_value\t\t %d\n' % int(self.nodata) )
            else:
                Output.write( 'NODATA_value\t\t %f\n' % self.nodata )
            for row in range( self.nrows-1,-1,-1 ):
                record = []
                for col in range( self.ncols ):
                    if Integer:
                        record.append( str( int( round( self.data[row,col]) ) ) )
                    else:
                        record.append( str(self.data[row,col]) )
                Output.write( string.join(record, ' ')+'\n' )
            Output.close()
        except:
            print "Error writing grid ::", self.name

    def copy( self ):
        """return a copy of a grid"""
        New = grid(self.data, self.xllcorner, self.yllcorner, self.cellsize, 'copy-'+self.name, self.nodata)
        return New

    def ApplyMask(self, A ):
        if type(A)==type(self):
            mask = A.data
        else:
            mask = A
        self.data = numpy.where( mask == 1, self.data, self.nodata)

    def new( self, d, x, y, dx, n='temp.grd', nd=-999.0):
        """Create a new grid from a numpy array, (x,y) location of lower left corner and grid size. Optional named parameters include n=name and nd=nodata value."""
        self.data  = d
        self.name = n
        self.ncols = self.data.shape[1]
        self.nrows = self.data.shape[0]
        self.xllcorner = x
        self.yllcorner = y
        self.cellsize  = dx
        self.nodata    = nd


    def ListColorMaps(self):
        """Utility function to display matplotlib color maps as an image"""
        p.rc('text', usetex=False)
        a=p.outerproduct(numpy.arange(0,1,0.01),numpy.ones(10))
        p.figure(figsize=(10,5))
        p.subplots_adjust(top=0.8,bottom=0.05,left=0.01,right=0.99)
        maps=[m for m in p.cm.datad.keys() if not m.endswith("_r")]
        maps.sort()
        l=len(maps)+1
        i=1
        for m in maps:
            p.subplot(1,l,i)
            p.axis("off")
            p.imshow(a,aspect='auto',cmap=p.get_cmap(m),origin="lower")
            p.title(m,rotation=90,fontsize=10)
            i=i+1
        #savefig("colormaps.png",dpi=100,facecolor='gray')
        p.show()

    def draw(self, **kwargs):
        """Create plots of ascii grid using matplotlib.
           Keyword arguements
           contours = if integer then number of contours else list of contour values. default is 10
           cmap = specify matplotlib color map to use. default=jet
           dmap = map features to draw. default=4
                = 0, draw nothing
                = 1, draw parallels and meridians
                = 2, draw coastlines
                = 3, draw countries
                = 4, draw states
           res = resolution of matplotlib map: lo (default), med, hi.
           shapefiles = comma separated list of tuples for shapefile attributes.
                        [('filename', 'desc', True/False, linewidth=0.5, color='k'->colors are matplot lib line colors)]
            title = plot title - defaults to ascii grid name attribute
            format = output format (ps, png)
            dpi = dots per inch for output formats that support it
        """

        Lons = numpy.ones(self.data.shape)*0.5
        Lats = numpy.ones(self.data.shape)*0.5
        for ix in range(self.ncols):
            for iy in range(self.nrows):
                Lons[iy,ix] = self.xllcorner+float(ix)*self.cellsize
                Lats[iy,ix] = self.yllcorner+float(iy)*self.cellsize
        ContourMin = numpy.min(numpy.where(self.data != self.nodata,self.data, 1000000))
        ContourMax = numpy.max(numpy.where(self.data != self.nodata,self.data, -1000000))*1.10
        if kwargs.has_key('contours'):
            if type( kwargs['contours'] ) == type( 1 ):
                Contours = numpy.arange(ContourMin, ContourMax, (ContourMax-ContourMin)/float( kwargs['contours']+1))
            else:
                Contours = kwargs['contours']
        else:
            Contours = numpy.arange(ContourMin, ContourMax, (ContourMax-ContourMin)/11.)
        if kwargs.has_key('cmap'):
            mycmap = kwargs['cmap']
        else:
            mycmap = 'jet'
        if kwargs.has_key('dmap'):
            dmap = max(0,min(4,kwargs['dmap']))
        else:
            dmap = 4
        # Lambert Conformal Conic map.
        if kwargs.has_key('res'):
            if kwargs['res']=='med':
                mapres='i'
            elif kwargs['res']=='hi':
                mapres='h'
            else:
                mapres = 'l'
        else:
            mapres = 'l'
        if mapres not in ('c','l','i','h'):
            mapres = 'l'
        m = Basemap(llcrnrlon=Lons[0,0], llcrnrlat=Lats[0,0], urcrnrlon=Lons[self.nrows-1,self.ncols-1], urcrnrlat=Lats[self.nrows-1,self.ncols-1],
                    projection='lcc',lat_1=30.,lat_2=60.,lon_0=(Lons[0,0]+Lons[self.nrows-1,self.ncols-1])/2.,
                    resolution =mapres,area_thresh=1000.)
        # create figure, add axes.
        fig=p.figure()
        ax = fig.add_axes([0.1,0.1,0.7,0.7])
        #make a filled contour plot.
        x, y = m( Lons , Lats)
        CS = m.contourf(x,y,self.data, Contours, cmap=p.get_cmap(mycmap))
	pos = ax.get_position()
	l, b, w, h = getattr(pos, 'bounds', pos)
        #l,b,w,h=ax.get_position()
        cax = p.axes([l+w+0.075, b, 0.05, h]) # setup colorbar axes
        p.colorbar(drawedges=True, cax=cax) # draw colorbar
        p.axes(ax)  # make the original axes current again

        if kwargs.has_key('shapefiles'):
            for s in kwargs['shapefiles']:
                try:
                    lw = s[3]
                except:
                    lw = 0.5
                try:
                    clr = s[4]
                except:
                    clr='k'
                shp_info = apply(m.readshapefile, (s[0],s[1]),{'drawbounds':s[2], 'linewidth':lw, 'color':clr} )
        # draw coastlines, meridians and parallels.
        if dmap > 1:
            m.drawcoastlines()
        if dmap > 2:
            m.drawcountries()
        if dmap > 3:
            m.drawstates()
        if dmap > 0:
            m.drawparallels(p.arange(10,70,10),labels=[1,1,0,0])
            m.drawmeridians(p.arange(-100,0,10),labels=[0,0,0,1])
        if kwargs.has_key('title'):
            p.title(kwargs['title'])
        else:
            p.title(self.name.title())
        if kwargs.has_key('format'):
            fn = self.name+'.'+kwargs['format']
            if kwargs.has_key('dpi'):
                dots = kwargs['dpi']
            else:
                dots = 100
            try:
                p.savefig(fn,dpi=dots)
            except:
                print 'Error saving to format : ', kwargs['format']
        else:
            p.show()

    def extract(self, xll, yll, xur, yur, n=''):
        """Extract a subsection of a grid and return it as a new grid instance"""
        #find bounds of selection
        ixstart   = min(max( int( (float(xll)-self.xllcorner)/self.cellsize ), 0), self.ncols)
        ixend     = min(max( int( (float(xur)-self.xllcorner)/self.cellsize ), ixstart), self.ncols)
        iystart   = min(max( int( (float(yll)-self.yllcorner)/self.cellsize ), 0), self.nrows)
        iyend     = min(max( int( (float(yur)-self.yllcorner)/self.cellsize ), iystart), self.nrows)
        region  = numpy.copy( self.data[iystart:iyend,ixstart:ixend] )
        x = self.xllcorner + float(ixstart) * self.cellsize
        y = self.yllcorner + float(iystart) * self.cellsize
        if n == '':
            parts = self.name.split('.')
            n = parts[0]+'-subregion.'+parts[1]
        New = grid(region, x, y, self.cellsize, n, self.nodata)
        return New

    def header(self):
        """Print ascii grid header in formation"""
        print 'dimensions',self.data.shape
        print 'llcorner', self.xllcorner, self.yllcorner
        print 'cell size', self.cellsize

    def toKML(self, name='temp', **kwargs):
        if kwargs.has_key('cmap'):
            mycmap = kwargs['cmap']
        else:
            mycmap = 'jet'
        figsize=(numpy.array(self.data.shape)/100.0)[::-1]
        p.rcParams.update({'figure.figsize':figsize})
        fig = p.figure(figsize=figsize)
        p.axes([0,0,1,1])
        p.axis('off')
        fig.set_size_inches(figsize)
        p.imshow(self.data, origin='lower', cmap=p.get_cmap(mycmap))
        p.savefig(name+'.png', facecolor='black', edgecolor='black', dpi=100)
        ImgFileName = name+'.png'
        Left = self.xllcorner
        Right= Left + (self.ncols-1) * self.cellsize
        Bottom = self.yllcorner
        Top = Bottom + (self.nrows-1) * self.cellsize
        kml = '''<?xml version="1.0" encoding="UTF-8"?>
                    <kml xmlns="http://earth.google.com/kml/2.0">
                    <Document>
                    <GroundOverlay>
          <name>%s</name>
          <drawOrder>1</drawOrder>
          <Icon>
            <href>%s</href>
          </Icon>
          <altitude>0</altitude>
          <altitudeMode>clampToGround</altitudeMode>
          <LatLonBox>
            <north>%f</north>
            <south>%f</south>
            <east>%f</east>
            <west>%f</west>
            <rotation>0</rotation>
          </LatLonBox>
          </GroundOverlay>\n</Document>\n</kml>''' % (name, ImgFileName, Top, Bottom, Right, Left)
        kmlfile = open(name+'.kml', 'w')
        kmlfile.write(kml)
        kmlfile.close()

def griddata( X, Y, Z, xl, yl, xr, yr, dx):
    # define grid.
    xi, yi = p.meshgrid( p.linspace(xl,xr, int((xr-xl)/dx)+1), p.linspace(yl,yr, int((yr-yl)/dx)+1))
    # grid the data.
    zi = mgriddata(X,Y,Z,xi,yi)
    New = grid( zi, xl, yl, dx)
    return New

def AvgGrids( A ):
    """Calculate the average of a list of grids and return a grid with the average and a list of anomaly grids."""
    Sum   = numpy.zeros( A[0].data.shape ) * 1.
    Count = numpy.zeros( A[0].data.shape ) * 1.
    for a in A:
        Count = Count+numpy.where(a.data != a.nodata, 1., 0.)
        Sum   = Sum+numpy.where(a.data != a.nodata, a.data, 0.)
    Mean = numpy.where( Count>0, Sum/Count, A[0].nodata )
    Anom = []
    for a in A:
        b = a.copy()
        b.data = numpy.where(b.data != b.nodata, b.data-Mean, b.nodata)
        b.name = 'anom-'+b.name
        Anom.append( b )
    New = grid(Mean, A[0].xllcorner, A[0].yllcorner, A[0].cellsize, 'mean.grd', A[0].nodata)
    return New, Anom


def AddGrids(A, B):
    """Input two ascii grids, make sure bounds match and return new grid"""
    if (A.xllcorner,A.yllcorner) == (B.xllcorner,B.yllcorner) and (A.ncols,A.nrows)==(B.ncols,B.nrows):
        maxVal = max( numpy.max(A.data), numpy.max(B.data))
        Ax = numpy.where(A.data != A.nodata, A.data+maxVal, 0.0)
        Bx = numpy.where(B.data != B.nodata, B.data+maxVal, 0.0)
        C = Ax+Bx
        C = numpy.where(C != 0.0, C-2.*maxVal, 0.0)
        C = numpy.where(C < 0.0, C+maxVal, C)
        C = numpy.where(C != 0.0, C, A.nodata)
        New = grid(C, A.xllcorner, A.yllcorner, A.cellsize, 'sum.grd', A.nodata)
        return New
    else:
        return "Error: grid mismatch"

def SubtractGrids(A, B):
    """Input two ascii grids, make sure bounds match and return new grid"""
    if (A.xllcorner,A.yllcorner) == (B.xllcorner,B.yllcorner) and (A.ncols,A.nrows)==(B.ncols,B.nrows):
        maxVal = max( numpy.max(A.data), numpy.max(B.data))
        Ax = numpy.where(A.data != A.nodata, A.data+maxVal, 0.0)
        Bx = numpy.where(B.data != B.nodata, B.data+maxVal, 0.0)
        C = A.data - B.data
        #C = numpy.where(C != 0.0, C-2.*maxVal, 0.0)
        #C = numpy.where(C < 0.0, C+maxVal, C)
        #C = numpy.where(C != 0.0, C, A.nodata)
        New = grid(C, A.xllcorner, A.yllcorner, A.cellsize, 'subtract.grd', A.nodata)
        return New
    else:
        return "Error: grid mismatch"

def PDiffGrids(A, B):
    """Input two ascii grids, make sure bounds match and return new grid"""
    if (A.xllcorner,A.yllcorner) == (B.xllcorner,B.yllcorner) and (A.ncols,A.nrows)==(B.ncols,B.nrows):
        Bx = numpy.where(B.data != B.nodata, B.data, 1.0)
        Bx = numpy.where(B.data != 0., B.data, 1.0)
        C = 100. * (A.data-Bx)/Bx
        New = grid(C, A.xllcorner, A.yllcorner, A.cellsize, 'pdif.grd', A.nodata)
        return New
    else:
        return "Error: grid mismatch"

def QueryGrid(A, val, op, **kwargs):
    """Query grid for val based on op( ==, <, <=, !=, >, >=). Returns a new grid.
       keyword options for true and false values. Use to create a mask by setting true=1 and false=0"""
    if kwargs.has_key('true'):
        T=kwargs['true']
    else:
        T=A.data
    if kwargs.has_key('false'):
        F=kwargs['false']
    else:
        F=A.nodata
    Q = eval('A.data'+op+str(val))
    C = numpy.where( Q, T, F)
    New = grid(C, A.xllcorner, A.yllcorner, A.cellsize, 'query.grd', A.nodata)
    return New

if __name__ == "__main__":
    ValidGridFile = 'eastcoast-topo2.grd'
    InvalidGridFile = 'MeanWind.grd'
    #Test module
    # Read valid grid
    Failed = 0
    try:
        topo = grid( ValidGridFile )
    except:
        print 'Fail - Read valid grid'
        Failed += 1
    else:
        print 'Pass - Read valid grid'
    # Read invalid grid
    try:
        test = grid( InvalidGridFile )
    except:
        print 'Pass - Read invalid grid'
    else:
        print 'Fail - Read invalid grid'
        Failed += 1
    try:
        topo.draw()
    except:
        print "Fail - drawing map"
        Failed += 1
    else:
        print "Pass - drawing map"
    try:
        landmask = QueryGrid(topo, 0.0, '>=', true=1, false=0)
    except:
        print "Fail - grid query"
        Failed += 1
    else:
        print "Pass - grid query"
    try:
        topo.ApplyMask( landmask )
    except:
        print "Fail - apply mask"
        Failed += 1
    else:
        topo.draw()
        print "Pass - apply mask"
    try:
        region = topo.extract(-90.0,25.0,-80.0,35.0)
    except:
        print "Fail - extract region"
        Failed += 1
    else:
        print "Pass - extract region"
    try:
        region.write()
        region2=grid( region.name )
    except:
        print "Fail - write/read grid"
        Failed += 1
    else:
        region2.draw()
        print "Pass - write/read grid"
    try:
        topo.toKML()
    except:
        print "Fail - output kml"
        Failed += 1
    else:
        print "Pass - output kml"
    try:
        xl =-90.0
        yl =25.0
        xr=-80.0
        yr=35.0
        dx=topo.cellsize
        X=[]
        Y=[]
        Z=[]
        for i in range(100):
           x = random.random()*(xr-xl)+xl
           y = random.random()*(yr-yl)+yl
           z = random.random() *100.
           X.append(x)
           Y.append(y)
           Z.append(z)
        G = griddata( numpy.array(X),numpy.array(Y),numpy.array(Z), xl,yl,xr,yr,dx)
        G.draw()
    except:
        print "Fail - irregular data"
        Failed += 1
    else:
        print "Pass - irregular data"
    print 'Number of tests failed = ', Failed

