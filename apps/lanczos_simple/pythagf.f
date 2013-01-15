      real function pythagf(a,b)
      real a,b
c
c     finds dsqrt(a**2+b**2) without overflow or destructive underflow
c
      real p,r,s,t,u
      p = amax1(abs(a),abs(b))
      if (p .eq. 0.0e0) go to 20
      r = (amin1(abs(a),abs(b))/p)**2
   10 continue
         t = 4.0e0 + r
         if (t .eq. 4.0e0) go to 20
         s = r/t
         u = 1.0e0 + 2.0e0*s
         p = u*p
         r = (s/u)**2 * r
      go to 10
   20 pythagf = p
      return
      end
