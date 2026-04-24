import { type NextRequest, NextResponse } from "next/server";
import { SignJWT } from "jose";
import { SESSION_COOKIE, COOKIE_OPTIONS } from "@/lib/auth/cookies";

const DEMO_PASSWORD = process.env.SAS_DEMO_PASSWORD;
const JWT_SECRET = new TextEncoder().encode(
  process.env.SAS_JWT_SECRET ?? "dev-secret"
);

export async function POST(req: NextRequest): Promise<NextResponse> {
  let body: { username?: string; password?: string };
  try {
    body = await req.json();
  } catch {
    return NextResponse.json(
      { error: "Invalid request body." },
      { status: 400 }
    );
  }

  const { password } = body;

  if (!password) {
    return NextResponse.json(
      { error: "Password is required." },
      { status: 400 }
    );
  }

  // If SAS_DEMO_PASSWORD is unset, any non-empty password succeeds (dev bypass).
  const isValid =
    DEMO_PASSWORD === undefined || DEMO_PASSWORD === ""
      ? password.length > 0
      : password === DEMO_PASSWORD;

  if (!isValid) {
    return NextResponse.json(
      { error: "Invalid credentials. Please try again." },
      { status: 401 }
    );
  }

  const now = Math.floor(Date.now() / 1000);
  const token = await new SignJWT({ sub: "demo" })
    .setProtectedHeader({ alg: "HS256" })
    .setIssuedAt(now)
    .setExpirationTime(now + 86400) // 24 hours
    .sign(JWT_SECRET);

  const response = NextResponse.json({ ok: true }, { status: 200 });
  response.cookies.set(SESSION_COOKIE, token, COOKIE_OPTIONS);
  return response;
}
