import { NextResponse } from 'next/server';
import connectToDatabase from '@/lib/mongodb';
import User from '@/models/User';
import Screening from '@/models/Screening';
import { mockStore } from '@/lib/mockStore';

export async function GET() {
  try {
    const db = await connectToDatabase();

    if (db) {
      const users = await User.find({}).select('-password').sort({ createdAt: -1 }).lean();
      const totalScreenings = await Screening.countDocuments({});
      const highRiskScreenings = await Screening.countDocuments({ isDetected: true });

      return NextResponse.json({
        success: true,
        users,
        stats: {
          totalUsers: users.length,
          totalScreenings,
          highRiskScreenings,
          systemStatus: "Operational (MongoDB Atlas Live)",
        },
      });
    } else {
      const users = mockStore.getUsers().map(({ passwordHash, ...rest }) => rest);
      const allScr = mockStore.getScreenings();
      const highRisk = allScr.filter((s) => s.isDetected).length;

      return NextResponse.json({
        success: true,
        users,
        stats: {
          totalUsers: users.length,
          totalScreenings: allScr.length,
          highRiskScreenings: highRisk,
          systemStatus: "Operational (Local / In-Memory Demo)",
        },
      });
    }
  } catch (error) {
    console.error('Admin users fetch error:', error);
    return NextResponse.json({ error: error.message }, { status: 500 });
  }
}
