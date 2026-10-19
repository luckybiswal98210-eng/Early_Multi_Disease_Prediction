import { NextResponse } from 'next/server';
import connectToDatabase from '@/lib/mongodb';
import Screening from '@/models/Screening';
import { mockStore } from '@/lib/mockStore';

export async function GET(req) {
  try {
    const { searchParams } = new URL(req.url);
    const userId = searchParams.get('userId');
    const role = searchParams.get('role');

    const db = await connectToDatabase();

    if (db) {
      const filter = (role === 'admin' && !userId) ? {} : (userId ? { userId } : {});
      const screenings = await Screening.find(filter).sort({ timestamp: -1 }).lean();
      return NextResponse.json({ success: true, screenings });
    } else {
      const filterId = (role === 'admin' && !userId) ? null : userId;
      const screenings = mockStore.getScreenings(filterId);
      return NextResponse.json({ success: true, screenings });
    }
  } catch (error) {
    console.error('Fetch screenings error:', error);
    return NextResponse.json({ error: error.message }, { status: 500 });
  }
}

export async function POST(req) {
  try {
    const data = await req.json();
    const { userId, userName, userEmail, disease, probability, status, vitals, notes } = data;

    if (!userId || !disease || probability === undefined) {
      return NextResponse.json({ error: 'Missing required screening fields.' }, { status: 400 });
    }

    const isDetected = Number(probability) >= (disease.includes("Parkinson") ? 80 : 50);

    const db = await connectToDatabase();

    if (db) {
      const newScreening = await Screening.create({
        userId,
        userName: userName || 'Anonymous Patient',
        userEmail: userEmail || 'patient@health.org',
        disease,
        probability: Number(probability),
        status,
        isDetected,
        vitals: vitals || '',
        notes: notes || '',
        timestamp: new Date(),
      });
      return NextResponse.json({ success: true, screening: newScreening });
    } else {
      const newScreening = mockStore.addScreening({
        _id: `scr_${Date.now()}`,
        userId,
        userName: userName || 'Anonymous Patient',
        userEmail: userEmail || 'patient@health.org',
        disease,
        probability: Number(probability),
        status,
        isDetected,
        vitals: vitals || '',
        notes: notes || '',
        timestamp: new Date().toISOString(),
      });
      return NextResponse.json({ success: true, screening: newScreening });
    }
  } catch (error) {
    console.error('Save screening error:', error);
    return NextResponse.json({ error: error.message }, { status: 500 });
  }
}
