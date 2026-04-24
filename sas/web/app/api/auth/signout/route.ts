import { NextResponse } from "next/server";
import { SESSION_COOKIE } from "@/lib/auth/cookies";

export async function GET(): Promise<NextResponse> {
  const response = NextResponse.redirect(
    new URL("/signin", process.env.NEXT_PUBLIC_APP_URL ?? "http://localhost:3000")
  );
  response.cookies.set(SESSION_COOKIE, "", {
    httpOnly: true,
    maxAge: 0,
    path: "/",
  });
  return response;
}
