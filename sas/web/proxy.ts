import { NextRequest, NextResponse } from "next/server";
import { jwtVerify } from "jose";
import { SESSION_COOKIE } from "@/lib/auth/cookies";

const JWT_SECRET = new TextEncoder().encode(
  process.env.SAS_JWT_SECRET ?? "dev-secret"
);

// Paths that do not require authentication.
const PUBLIC_PATHS = [
  "/signin",
  "/api/signin",
  "/api/auth/signout",
];

function isPublicPath(pathname: string): boolean {
  return (
    PUBLIC_PATHS.some(
      (p) => pathname === p || pathname.startsWith(p + "/")
    ) ||
    pathname.startsWith("/_next/") ||
    pathname.startsWith("/favicon") ||
    pathname.startsWith("/static/")
  );
}

export async function proxy(req: NextRequest): Promise<NextResponse> {
  const { pathname } = req.nextUrl;

  if (isPublicPath(pathname)) {
    return NextResponse.next();
  }

  const token = req.cookies.get(SESSION_COOKIE)?.value;

  if (!token) {
    const signInUrl = req.nextUrl.clone();
    signInUrl.pathname = "/signin";
    return NextResponse.redirect(signInUrl);
  }

  try {
    await jwtVerify(token, JWT_SECRET);
    return NextResponse.next();
  } catch {
    // Token invalid or expired — clear the cookie and redirect to sign-in.
    const signInUrl = req.nextUrl.clone();
    signInUrl.pathname = "/signin";
    const response = NextResponse.redirect(signInUrl);
    response.cookies.set(SESSION_COOKIE, "", { maxAge: 0, path: "/" });
    return response;
  }
}

export const config = {
  // Run proxy on all paths except Next.js internals and static assets.
  matcher: ["/((?!_next/static|_next/image|favicon.ico).*)"],
};
